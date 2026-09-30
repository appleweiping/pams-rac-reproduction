# PAMS full multi-scale SSHead/readout failure diagnosis

**Status:** development-only, read-only diagnosis
**Scope:** two already-scored strict 337-train full multi-scale runs, seeds
`2026` and `42`
**Test boundary:** untouched

This audit did not train a model, run a new prediction, select a threshold, or
inspect the 105-video test split. Development targets were opened only for
post-hoc error analysis of prediction artifacts that had already been frozen
by the target-isolated predictor.

## Finding

The `NMAE ~= 4.7` failure is a structural SSHead/readout collapse onto
absolute position, not an original-fps-to-256-frame conversion error.

Exactly 43 of the 84 development pose caches contain 256 valid frames. In
both seeds, every one of those 43 videos has `period_frames = 8` and
`count = 31`, despite spanning different actions and ground-truth counts.
Their centered SSHead streams are almost identical:

| Quantity | Seed 2026 | Seed 42 |
|---|---:|---:|
| Full-valid videos with period 8 | 43/43 | 43/43 |
| Full-valid videos with count 31 | 43/43 | 43/43 |
| Median pairwise stream correlation | 0.998712 | 0.998926 |
| Correlation: prediction vs target | 0.0730 | 0.0663 |
| Correlation: prediction vs valid frames | 0.9924 | 0.9895 |

The model is effectively counting an eight-frame positional waveform for as
long as the mask remains valid. Its prediction differs from
`valid_frames / 8` by only `1.63` counts on average in seed 2026 and `1.67`
in seed 42.

## Aggregate evidence

Ground-truth count has median `5`, mean `5.94`, and range `2-29`. Both runs
instead have predicted median `31`, mean about `22.7`, and maximum `31`.

| Quantity | Seed 2026 | Seed 42 |
|---|---:|---:|
| NMAE / OBO | 4.720314 / 0.059524 | 4.741497 / 0.035714 |
| Period exactly 8 | 72/84 | 72/84 |
| Count exactly 31 | 44/84 | 44/84 |
| Raw stream FFT maximum at bin 32 | 50/84 | 53/84 |
| Fast equals medium expert | 68/84 | 65/84 |
| Majority vote / FFT fallback | 70/14 | 67/17 |
| Selected slow expert | 0 | 0 |

The failure is seed-stable: `63/84` counts and `78/84` periods match
exactly; count correlation is `0.9979` and mean absolute seed-to-seed count
difference is `0.345`.

Action grouping follows pose validity and the positional template rather than
action periodicity. In seed 2026, JumpingJack has target mean `2.5` but
prediction mean `31` and NMAE `11.917`; BodyWeightSquats has target mean
`2.75`, prediction mean `31`, and NMAE `10.625`. Pose-missing groups such as
BreastStroke have much lower prediction means. Five videos are fully invalid
and all predict zero. Those failures matter, but cannot explain why every
fully valid video predicts 31.

## Why position encoding is the primary source

The default path adds sinusoidal position encoding before the Transformer
(`src/pams/model.py:170-177`), then estimates the post-warmup pseudo-period
from the highest-variance coordinate of embedding velocity
(`src/pams/period.py:59-103`,
`src/pams/training.py:341-355`). The resulting embeddings already contain
absolute-frame oscillations.

An earlier read-only period-source diagnostic found an embedding period of
`6.736842 = 256 / 38` on `63/84` videos. Its valid-duration floor counter had
NMAE `5.9042`, whereas the pose-source equivalent had NMAE `0.6097`. The
fastest standard sinusoidal positional components have a period near
`2*pi`, consistent with this fixed high-frequency bin. A prior inferred
pre-PE projected-vector variant improved SSHead NMAE to `2.0742`, still far
outside the frozen gate but supporting the direction of causality.

## Why the inferred SSHead does not correct it

SSHead training obtains its target period from embedding/projected evidence,
but inference estimates the period again from the scalar head stream
(`src/pams/training.py:2030-2071,2182-2225`). The independently inferred loss
does not provide a hard anti-collapse constraint. At epoch 30:

| Quantity | Seed 2026 | Seed 42 |
|---|---:|---:|
| Mean training stream std | 0.00772 | 0.00980 |
| Variance loss | 0.99228 | 0.99020 |
| Spectral loss | 0.24488 | 0.23816 |
| Cycle loss | 0.000008 | 0.000013 |

Thus the intended “standard deviation at least 1” behavior is not achieved:
the `0.1`-weighted variance hinge trades off against other losses and leaves
the head two orders of magnitude below unit scale.

Small amplitude is evidence that the inferred objective is underconstrained,
but it is not by itself the peak multiplier. The counter's height threshold
is a local mean plus a standard-deviation multiple, and prominence is a
fraction of stream range (`src/pams/consensus.py:104-130,257-315`). These are
approximately invariant to positive affine rescaling. The multiplier comes
from the waveform's fixed bin-32 frequency; two experts then agree on about
31 peaks, so majority voting wins before the FFT fallback.

## Why fps conversion is not the cause

Uniform resampling maps coordinates and masks into the same 256-frame domain
and adjusts fps metadata (`src/pams/data.py:1504-1541,1569-1588`). Prediction
never reads fps. The head stream, period estimator, peak distances, and masks
all use resampled-frame coordinates. A learned period of eight frames
therefore yields roughly `256 / 8 = 32` cycles and 31 interior peaks.
Converting the period to original-video frames and back cannot change the
count.

There is a secondary protocol mismatch: preprocessing trims the first-to-last
detected span rather than selecting a longest continuous valid track
(`src/pams/pose.py:221-251`). It changes valid duration and explains some
partial-mask variation, but the 43/43 full-valid collapse excludes it as the
primary high-frequency error.

## Frozen next experiment

First add a label-free period-source gate using static poses, time-shuffled
poses, and synthetic periodic poses with known counts `2-40`. Freeze the gate
before reading development metrics: static input should have near-zero period
confidence, the period histogram should not concentrate at 6-8 frames, and
synthetic periods should be recovered.

Then run one strict matched seed-2026 experiment changing exactly one
configuration field:

```yaml
period:
  post_warmup_source: projected_pose_velocity_vector_acf
```

All other 337-train/84-dev full multi-scale settings, SSHead loss, consensus,
data, and provenance requirements remain fixed. Score development once after
the label-free gate passes. Do not tune peak height, prominence, expert
distance, or fps conversion on the 84 labels; those parameters cannot repair
the positional waveform and would be new metric tuning.

All path-free hashes and aggregate measurements are recorded in
[`pams_full_multiscale_readout_diagnosis.json`](pams_full_multiscale_readout_diagnosis.json).
