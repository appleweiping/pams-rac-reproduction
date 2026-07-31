# Result status

**Verification status: `partial_reproduction`**

A designated-server component smoke has been recorded by the operator for the
current pose preprocessing revision, including a strict 526-video manifest
and identity-audited pose caches for all 421 official training videos. The
path-free audit, clean identity ledger, and digest summary are under
[`results/server-smoke`](server-smoke/README.md). Thirteen training caches are
all-invalid and remain masked rather than being dropped.

Three-seed training and evaluation on the fixed 84-video development split
have now produced negative evidence for PAMS-Literal and the independently
inferred PAMS-SSHead. Neither method meets the preregistered thresholds on
development data. The 105-video test split has **not** received predictions or
evaluation, and no test metric exists. The original f99 three-seed
development CLI did load and deserialize the complete 337/84/105 manifest, so
those historical artifacts are not proof that test-label bytes were never
parsed. The newer source revision `375d8da` adds committed label-free
train/dev/test-identity sidecars; its v3/v4 diagnostic commands did not
deserialize the complete labeled manifest.

This repository is therefore a **partial reproduction**, not a completed or
validated reproduction. Baselines and executable ablation variants remain
incomplete. The development results below must not be moved into a test
leaderboard.

The latest direct v8 readout diagnostics produced NMAE/OBO
`0.664627 / 0.261905` for PAMS-Literal and `0.529058 / 0.333333` for a
synthetic-frozen pose-only local-frequency readout. The latter is
paper-table-ineligible but shows that the pose cache retains usable periodic
signal and that the learned inferred SSHead/period stream is the primary
remaining bottleneck. Full target-isolated artifacts are under
[`dev-negative/pams_v8_frozen_readout_diagnostics_seed2026_07192de/`](dev-negative/pams_v8_frozen_readout_diagnostics_seed2026_07192de/).

The frozen-v8 `PAMS-TeacherPeriodDirect-inferred-v1` follow-up then applied
the projected-pose vector-ACF teacher directly, without training another
head. Its formal CLI replay matched all 84 frozen one-off predictions exactly
for raw count, rounded count, period, confidence, valid-frame count, video ID,
and video hash. The resulting NMAE/OBO is `2.860120 / 0.166667`, with
over-counting and harmonic aliases dominating the error. This is a negative,
post-hoc development diagnostic and cannot enter a paper table or authorize
seeds `42/3407` or test105. Full gate, prediction, evaluation, recovery, and
equivalence evidence is under
[`dev-negative/pams_teacher_period_direct_inferred_v9_seed2026_07192de/`](dev-negative/pams_teacher_period_direct_inferred_v9_seed2026_07192de/).

## UCFRep-526 development-only negative result

All rows use the same fixed 84-video development split and source revision
`f99f3fda91b8bf2c25fc75a42814d9c97df71ae0`. Standard deviations are sample
standard deviations across the three seeds. The thresholds are shown only as
a preregistered diagnostic; passing them on development data would not verify
the reproduction.

| Variant | Seed | NMAE | Raw MAE | RMSE | OBO |
|---|---:|---:|---:|---:|---:|
| PAMS-Literal | 42 | 0.638439 | 3.738095 | 5.387375 | 0.285714 |
| PAMS-Literal | 2026 | 0.641446 | 3.726190 | 5.449989 | 0.309524 |
| PAMS-Literal | 3407 | 0.909889 | 5.464286 | 7.378379 | 0.154762 |
| PAMS-Literal | mean ± sample SD | 0.729925 ± 0.155861 | 4.309524 ± 1.000071 | 6.071914 ± 1.131865 | 0.250000 ± 0.083333 |
| PAMS-SSHead (inferred) | 42 | 5.225998 | 20.000000 | 22.813425 | 0.047619 |
| PAMS-SSHead (inferred) | 2026 | 5.008007 | 19.154762 | 21.787666 | 0.035714 |
| PAMS-SSHead (inferred) | 3407 | 4.589608 | 17.476190 | 19.926054 | 0.071429 |
| PAMS-SSHead (inferred) | mean ± sample SD | 4.941204 ± 0.323411 | 18.876984 ± 1.284630 | 21.509048 ± 1.463711 | 0.051587 ± 0.018185 |

Both variants pass the `NMAE <= 0.228` and `OBO >= 0.666` diagnostic on
`0/3` individual seeds; neither three-seed mean passes. The aggregate's
10,000-sample paired bootstrap for `PAMS-SSHead - PAMS-Literal` gives an NMAE
difference 95% CI of `[+3.333701, +5.141728]` and an OBO difference 95% CI of
`[-0.269841, -0.130952]`, so the inferred head is materially worse in this
development run.

The compact path-free record, source aggregate digest, and two seed-2026
diagnostics are under
[`results/dev-negative`](dev-negative/README.md).
That directory also records the failed, independently inferred PE-scale v2
seed-2026 diagnostic. It weakened one positional shortcut indicator but did
not improve the literal-head development result and remained far outside the
acceptance thresholds.

A new seed-2026 single-scale representation diagnostic used only 337
label-free training videos, completed 150 encoder plus 30 inferred-SSHead
epochs, and was evaluated through the target-isolated predictor/scorer. It
obtained NMAE `4.624484` and OBO `0.047619`, again far outside the gate.
Its now-completed protocol-matched full multi-scale comparator obtained NMAE
`4.720314` and OBO `0.059524`. The multi-scale run is worse on NMAE and only
slightly better on OBO, so the pair does not support a clear multi-scale
advantage. A later independently inferred Medium-only switch reused that
exact full-model checkpoint and obtained NMAE/OBO
`4.690327 / 0.059524`. Multi-expert consensus therefore has worse NMAE by
`+0.029987` (paired 95% CI `[+0.002977, +0.064290]`) and identical OBO.
This also fails the preregistered multi-expert-over-single-expert condition.
It is not a paper Table 2 reconstruction because the paper does not uniquely
disclose the single-expert replacement. Exact receipts and hashes are in
[`pams_single_scale_seed2026_strict.json`](dev-negative/pams_single_scale_seed2026_strict.json)
and
[`pams_multiscale_seed2026_strict.json`](dev-negative/pams_multiscale_seed2026_strict.json),
with the expert comparison in
[`pams_medium_only_seed2026_strict.json`](dev-negative/pams_medium_only_seed2026_strict.json).
All rows are `table2_eligible=false`; the 105-video test split remains
untouched.

An independently frozen fixed-period-16, single-scale training proxy has now
also been retrained and scored at seed 2026. The paper does not disclose its
conventional-TCC baseline window or complete geometry, so this proxy cannot
claim the paper's Table 2 row. Fixed-period Literal obtained NMAE/OBO
`3.099083 / 0.095238`, versus `3.313016 / 0.083333` for the current-code
adaptive single-scale Literal rerun. Fixed-period inferred SSHead obtained
`1.705373 / 0.071429`, versus `4.624484 / 0.047619` for adaptive
single-scale. The paired adaptive-minus-fixed SSHead NMAE difference is
`+2.919111`, with 95% CI `[+2.365327, +3.486879]`. The observed direction
therefore does not support a period-adaptive advantage under this inferred
closure, and neither fixed row passes the gate. Full evidence is in
[`pams_fixed_period16_inferred_seed2026_strict.json`](dev-negative/pams_fixed_period16_inferred_seed2026_strict.json).

The fixed-period proxy has now completed all three preregistered seeds.
Literal obtained mean NMAE/OBO `1.555686 / 0.246032`, with sample standard
deviations `1.345183 / 0.131133`; inferred SSHead obtained
`1.755943 / 0.075397`, with sample standard deviations
`0.156771 / 0.006873`. Both pass the joint gate on `0/3` seeds. A
video-cluster bootstrap finds that SSHead significantly worsens NMAE and OBO
relative to Literal, despite lowering mean RMSE. Full evidence is in
[`pams_fixed_period16_inferred_three_seed_strict.json`](dev-negative/pams_fixed_period16_inferred_three_seed_strict.json).

The six-row, three-seed Table 2 proxy matrix is now closed on dev84. Every
row passes the frozen joint gate on `0/3` seeds; the best mean NMAE is
`1.132350 ± 0.883825` for the multi-scale Medium-only Literal proxy, while
the independently inferred Full completion obtains
`5.022471 ± 0.505117 / 0.051587 ± 0.013746` NMAE/OBO and is not its own
ablation best. Paired stratified bootstrap supports an aggregate NMAE gain
from multi-scale, but not an OBO gain, and multi-expert consensus worsens
NMAE. This is a negative development diagnostic, not a paper Table 2 or
test105 result. Full evidence is in
[`pams_table2_three_seed_strict.json`](dev-negative/pams_table2_three_seed_strict.json).

The preregistered pose-cache stress matrix is also complete for the full
multi-scale seed-2026 SSHead checkpoint. Among 11 clean/perturbed conditions,
the endpoint-preserving `2.0` first-half speed warp significantly worsens
NMAE by `+0.524400` and reduces OBO to zero. Other temporal and occlusion
changes have paired intervals crossing zero; geometric transforms change
almost no rounded counts. Because clean NMAE/OBO is already
`4.720314 / 0.059524`, this invariance is a failure diagnostic rather than
robustness evidence. Full results are in
[`pams_pose_stress_seed2026_strict.json`](dev-negative/pams_pose_stress_seed2026_strict.json).

The released PoseRAC-v1 checkpoint has also completed an oracle-free dev84
diagnostic. Exact `74/74` checkpoint restoration plus an inferred
dynamic-range channel selector obtains NMAE/OBO
`0.706178 / 0.238095`. This is ineligible for the original PoseRAC-v1 or PAMS
cells because the released evaluator's GT-count oracle is disabled, the
selector is independently inferred, and the input is the 256-frame PAMS
cache. Full evidence is in
[`poserac_v1_official_oracle_free_dev84_strict.json`](dev-negative/poserac_v1_official_oracle_free_dev84_strict.json).

The inferred PAMS-SSHead temporal-conv v7 repair has also completed a
post-hoc seed-3407 variance follow-up after seeds 42 and 2026 had already
failed. Seed 3407 obtains NMAE/OBO `0.740368 / 0.202381`; the three-seed mean
± sample SD is `0.723528 ± 0.025256 / 0.210317 ± 0.024782`, with `0/3`
seeds passing. The new run used separate target-free prediction and CPU
scoring containers and did not access test105. Because the third seed was
post-hoc and the repair is independently inferred, it cannot enter the paper
table or a verified claim. Full evidence is in
[`pams_sshead_temporal_conv_v7_three_seed_posthoc_6ee25a2/`](dev-negative/pams_sshead_temporal_conv_v7_three_seed_posthoc_6ee25a2/).

The independently inferred `JTSPS-count-only` baseline has now completed real
supervised train337/dev84 runs for seeds `42/2026/3407`. Mean ± sample SD
NMAE is `0.487634 ± 0.012843`, and OBO is
`0.408730 ± 0.018185`; the paired 10,000-sample 95% intervals for the
mean-across-seeds estimand are `[0.409718, 0.566997]` and
`[0.309524, 0.511905]`. This is not a random-weight smoke, but it is
permanently ineligible for the paper's JTSPS cell because the source protocol
does not uniquely disclose cycle-density supervision or decoding. The two
new train/predict containers had no development-target or test mount; separate
CPU scorers opened dev84 targets only after all predictions were frozen. Full
predictions, evaluations, three-seed statistics, source/runtime hashes, and
the negative claim boundary are in
[`jtsps_count_only_inferred_three_seed_9df2646/`](dev-negative/jtsps_count_only_inferred_three_seed_9df2646/).

The strict full multi-scale protocol has now also completed seeds `42`,
`2026`, and `3407` for both readouts. PAMS-Literal obtained mean NMAE/OBO
`1.489633 / 0.226190`; its NMAE population/sample standard deviations are
`1.175316 / 1.439463`, exposing substantial random-head seed sensitivity.
PAMS-SSHead obtained `5.022471 / 0.051587`; its NMAE population/sample
standard deviations are `0.412427 / 0.505117`. Each method passes the joint
gate on `0/3` seeds. The inferred SSHead worsens NMAE for every matched seed,
while OBO is worse for two seeds and improves by only `1/84` for seed 3407.
All predictions were frozen without a target mount and scored in separate
processes. Full confidence intervals and artifact bindings are in
[`pams_full_multiscale_three_seed_strict.json`](dev-negative/pams_full_multiscale_three_seed_strict.json).

The same directory now includes strict-firewall seed-2026 v3/v4 experiments.
Projected-vector period v3 obtained `3.966841 / 0.059524` with the literal
head and `2.074190 / 0.202381` with the inferred SSHead. Cross-scale-union v4
obtained `2.422484 / 0.130952` and `2.620294 / 0.107143`, respectively. All
four NMAE/OBO pairs fail the gate. Those separately inferred v3/v4 variants
remain seed-2026-only; the strict paper-config full multi-scale experiment
described above is the protocol that was expanded to seeds 42 and 3407.

A later strict 337-train follow-up changed only the post-warm-up period
source to a pre-position-encoding projected-pose vector ACF. Its label-free
static/shuffled/synthetic gate passed before scoring. The inferred SSHead
obtained NMAE/OBO `2.380005 / 0.095238`, versus
`4.720314 / 0.059524` for the matched default period source. The paired NMAE
difference was `-2.340309` with 95% CI
`[-3.109489, -1.639701]`; OBO's difference interval crossed zero. This
material improvement still fails the frozen reproduction gate. The
single-field structural diff, two excluded launch attempts, target-isolated
receipts, and complete hashes are in
[`pams_projected_pose_vector_acf_seed2026_strict.json`](dev-negative/pams_projected_pose_vector_acf_seed2026_strict.json).

A subsequent development-only local-frequency sweep found
`0.347761 / 0.511905` at its best-NMAE setting and
`0.361573 / 0.583333` at its best-OBO setting. These settings are explicitly
ineligible because development labels selected them, and neither passes the
gate. The exact paired-bootstrap intervals and artifact hashes are in
[`projected_vector_v3_v4_seed2026.json`](dev-negative/projected_vector_v3_v4_seed2026.json).

A separate synthetic-only frozen local-frequency readout obtained
`0.514566 / 0.333333` NMAE/OBO on the same 84-video development split. The
current official Google Research RepNet notebook and published `ckpt-70`
checkpoint were also run successfully on all 84 videos and obtained
`0.434505 / 0.583333`. Both fail the frozen gate and neither is eligible for
the original paper table. The RepNet row is explicitly an
`official-current-ckpt70` sanity under our independent dev protocol. Exact
confidence intervals and hashes are in
[`server_dev_readouts_repnet_4d4d708.json`](dev-negative/server_dev_readouts_repnet_4d4d708.json).

A successor local-frequency selector was then frozen without development
inputs, development targets, or any test105 asset. It selected one candidate
from the exact 512-member grid using train337 perturbation consistency,
synthetic count/stress accuracy, and a train-only non-degeneracy gate. Its
isolated dev84 run obtained NMAE/OBO `0.453804 / 0.369048` (95% bootstrap CIs
`[0.381032, 0.529153] / [0.261905, 0.476190]`). This improves the earlier
synthetic-only local-frequency NMAE but remains worse than both the official
RepNet and ESCounts development sanity rows, and `49/84` predictions equal
two. It therefore fails the frozen gate and remains a target-free-selected,
paper-table-ineligible negative diagnostic. Exact selector, predictions,
receipts, source-view audits, and hashes are under
[`pams_local_frequency_v2_target_free_dev84_2310d09/`](dev-negative/pams_local_frequency_v2_target_free_dev84_2310d09/).

The official TransRAC RepCount-A checkpoint was also restored with exact
230-key coverage and run on all 84 development videos in a modern
PyTorch/CUDA compatibility stack. It obtained rounded NMAE `0.690692` and
OBO `0.285714`; paired 10,000-sample 95% CIs are
`[0.562069, 0.848339]` and `[0.190476, 0.380952]`. This is a
development-only official-checkpoint sanity, not original evaluator parity.
The server artifact predates the strict public runner/scorer schema, so it is
retained as historical evidence. A clean current-code rerun at Git SHA
`b0b85f113a9ec752585eb0d054175ad50397c4cc` reproduced every raw and
rounded prediction exactly and emitted the same metrics.
Exact asset/runtime/receipt hashes are in
[`transrac_official_modern_compat_dev84.json`](dev-negative/transrac_official_modern_compat_dev84.json).

The official IVAC-P2L RepCount-A checkpoint likewise completed strict
label-free prediction and separate scoring on all 84 development videos.
The current half-up protocol obtained NMAE `0.666137` and OBO `0.333333`,
with 10,000-sample paired 95% CIs `[0.523205, 0.839972]` and
`[0.238095, 0.428571]`. All 84 raw model outputs exactly match the
historical adapter. One exact `8.5` output changes from legacy ties-to-even
rounding (`8`) to frozen half-up (`9`), so the current metrics are kept
separate. Exact bindings are in
[`ivac_p2l_official_dev84_retrospective.json`](dev-negative/ivac_p2l_official_dev84_retrospective.json).

The official ESCounts zero-shot path also completed its clean current
runner-to-scorer rerun at Git SHA
`bbaa7a1ee59bab022f8cc24f41a2a399bbd550e5`. Its 84 current raw and rounded
predictions are each exactly identical to the legacy ledger. The strict
scorer obtained NMAE `0.352419` and OBO `0.595238`, with paired 10,000-sample
95% CIs `[0.257909, 0.460455]` and `[0.488095, 0.702381]`. The primary/retry
flow records one accepted official-compatible tail shortfall (`669/670`
frames); larger shortfalls remain failures. Prediction had no target or test
mount, and development targets were opened only by the separate scorer.
Path-free full lineage and repair audit are in
[`escounts_official_dev84_f18fcf1.json`](dev-negative/escounts_official_dev84_f18fcf1.json).

The CountLLM-Lite resource gate was also executed in an isolated designated-
server container. One RTX A6000 exposed `47.318 GiB`, below the frozen
`48 GiB` minimum, so the status remains `smoke_only` and no full recipe was
started. The only CUDA work was a `70.7 MB` bridge forward/backward with
finite gradients. It did not load Vicuna, a video encoder, UCFRep samples,
development labels, or sealed-test inputs, and it is not a CountLLM baseline
result. The original artifact, receipt, hashes, and compact disclosure are in
[`results/resource-gates/countllm-lite-resource-smoke-59ae69f-20260729-v1`](resource-gates/countllm-lite-resource-smoke-59ae69f-20260729-v1/README.md).

## Standard UCFRep-526 fair table

| Method | Protocol | Status | NMAE | OBO |
|---|---|---|---:|---:|
| PAMS-Literal | 421/105 | dev negative available; test not run | — | — |
| PAMS-SSHead | 421/105 | dev negative available; test not run | — | — |
| RepNet | 421/105 | official-current ckpt-70 dev sanity available; original parity/test blocked | — | — |
| TransRAC | 421/105 | current-code modern-compat dev sanity available; original parity/test blocked | — | — |
| ESCounts | 421/105 | current-code official-checkpoint dev sanity available; original parity/test blocked | — | — |
| IVAC-P2L | 421/105 | current-code modern-compat dev sanity available; original parity/test blocked | — | — |
| PoseRAC-v1 official checkpoint | 421/105 | oracle-free dev84 diagnostic available; original protocol/test blocked | — | — |
| PoseRAC-ICONIP24 | 421/105 | implementation/data blocked | — | — |
| JTSPS-count-only | 421/105 | inferred clean-room three-seed dev84 negative available; source protocol/test blocked | — | — |
| CountLLM-Lite | 421/105, non-comparable recipe | 47.318 GiB < 48 GiB; bridge smoke-only / full adapter not run | — | — |

## UCFRep-pose-110 fair table

This table is a target only. Sealed scoring is disabled until the canonical
89/21 official annotation identity has been independently frozen.

| Method | Protocol | Status | NMAE | OBO |
|---|---|---|---:|---:|
| PAMS-Literal | 89/21 | data blocked | — | — |
| PAMS-SSHead | 89/21 | data blocked | — | — |
| PoseRAC-v1 fair rewrite | 89/21 | annotation/parity blocked | — | — |
| GMFL | 89/21 | annotation/parity blocked | — | — |
| SPKDB | 89/21 | annotation/parity blocked | — | — |
| BIGC | 89/21 | annotation/parity blocked | — | — |

The deterministic `spectral-proxy` is a synthetic pipeline diagnostic and is
never eligible for either paper-comparison table.

The current label-free local safety evidence, including the intentionally
failed SSHead collapse diagnostic, is published under
[`results/safety`](safety/README.md). Its designated-server rerun and real
UCFRep preprocessing smoke are summarized separately under
[`results/server-smoke`](server-smoke/README.md).
