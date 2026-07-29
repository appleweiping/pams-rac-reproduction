# Development-only negative evidence

**Status:** `partial_reproduction`

**Split:** fixed UCFRep-526 development split, 84 videos
**Source revision:** `f99f3fda91b8bf2c25fc75a42814d9c97df71ae0`

This directory records negative development evidence for the literal Period
Head and the independently inferred SSHead completion. It is not a test
result, leaderboard result, successful reproduction claim, or ablation
result.

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
