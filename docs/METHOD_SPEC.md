# PAMS implementation specification

This document maps the method in the
[CVPR Findings 2026 paper](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)
to the independent implementation. Values absent from the paper are linked to
[ASSUMPTIONS.md](ASSUMPTIONS.md) and are never presented as author settings.

## Inputs and notation

For a batch of normalized skeleton sequences,

```text
X: [B,T,33,3]
M: [B,T]       (true where a usable pose exists)
Z = Eθ(X): [B,T,512]
P = Hφ(Z): [B,T]
```

The paper defines the general input as `K×3`; `K=33`, MediaPipe extraction,
and per-frame min-max normalization are independent closures. Uniform
256-frame resampling is the historical closure used by v2/v3 and the existing
`official-segment-full-timeline-v1` caches. The current v3 preprocessing selects the earliest longest
contiguous main-subject detection run before normalization and resampling.
Invalid videos remain all zero and masked. Historical v2 results retain their
first-to-last-detection-span identity and are never overwritten by v3 caches.
The implementation does not claim cross-person identity tracking. A one-frame
v3 run is retained in the denominator as an all-invalid cache, and extraction
receipts record the selected source-run length.

The separately fingerprinted `official-segment-full-timeline-v1` protocol
uses each official UCFRep `.mat` file's 1-based inclusive `start_frame` and
`end_frame` only to define the input clip. The privileged manifest builder
converts that interval once to 0-based half-open
`[clip_start_frame, clip_end_frame)`. Count and `temporal_bound` cycle
locations never enter the label-free pose-input manifest. Pose extraction
seeks to `clip_start_frame`, decodes at most the exact interval length, keeps
every pose miss as an invalid frame, and resamples the complete clip timeline;
it never applies detected-span or longest-track trimming. See
[OFFICIAL_SEGMENT_PROTOCOL.md](OFFICIAL_SEGMENT_PROTOCOL.md) for schema and
coverage rules.

The separately fingerprinted recovery revision
`official-segment-heavy-missing-retry-full-timeline-v4a` retains the same
count-free interval but uses `temporal_resampling: none_native_timeline`.
Its output has exactly
`T = clip_end_frame - clip_start_frame`: decoded frames keep their original
interval-relative indices, an early-EOF suffix is exact-zero/mask-false, and
internal misses remain holes. Variable-length sequences are padded only to the
maximum `T` of the current batch; that temporary zero/mask padding does not
alter stored indices, FPS, or period units. This closure is supported by the
recovered standard UCFRep loader's frame-range/padding behavior and by strong
evidence in the PAMS supplement's long native-timeline plots, including
pause/activity structure around frames 800–1000. It is not claimed as exact
author code.

V4a currently authorizes only a paired, label-free train337 pose-input gate.
The inherited period bounds (`4–128`), position mode, and training block in
its experiment YAML have not been validated for native timelines. A passing
pose gate therefore does not authorize baseline training or establish a
native-timeline PAMS result; a dynamic-period repair must receive a new
configuration fingerprint before optimization.

## Encoder

`PAMSEncoder` implements the disclosed network:

1. flatten each frame to 99 values;
2. linear projection to 512;
3. add sinusoidal temporal position encoding;
4. apply four batch-first Transformer encoder layers, each with 16 heads,
   2,048 hidden feed-forward units, ReLU, 0.1 dropout and post layer norm;
5. L2-normalize every valid frame embedding.

Fully invalid sequences are guarded before attention to avoid all-masked
softmax NaNs, then returned as exact zeros.

The frozen configuration keeps `model.input_projection_scale: none`, which
is bitwise compatible with the original implementation. The separately
fingerprinted
[`pams_pe_scale_v2`](../configs/experiments/pams_pe_scale_v2.yaml)
diagnostic instead multiplies the linear projection by `sqrt(model_dim)`
before adding absolute positions. This is an inferred anti-shortcut repair,
not an author-disclosed setting, and it cannot enter the frozen sealed-test
table.

## Stop-gradient period estimate

The paper says the period is estimated by applying FFT to an autocorrelation
of stop-gradient embeddings or a pose-energy proxy. The historical/default
implementation:

1. during pose warm-up, selects the highest-variance signed pose coordinate;
   thereafter takes mask-aware first differences of detached embeddings and
   selects the highest-variance signed velocity coordinate (never an L2 speed,
   which can halve a sinusoid's apparent period);
2. mean-centres it over valid frames;
3. computes mask-normalized, non-circular autocorrelation with FFT;
4. applies a Hann window and finds dominant spectral power corresponding to
   periods from 4 through 128 frames.

The first ten encoder epochs use pose coordinates. Later epochs use detached
encoder embeddings. Period selection has no gradient.

The pre-dev-frozen v14 correction instead keeps the complete detached
embedding-velocity vector, computes signed cross-time dot products over all
512 dimensions with a mask-normalized vector autocorrelation, and applies the
same bounded spectral selector. This route is invariant to an orthogonal
change of embedding basis. It is an independently inferred interpretation of
the paper's underspecified "autocorrelation of embeddings," not a recovered
author implementation.

## PAMS TCC

For each anchor `z[t]`, local positives are valid `z[t-1]` and `z[t+1]`.
Past/future cycle positives are the highest-cosine-similarity valid frames in
the frozen correspondence neighbourhood around `t-T_hat` and `t+T_hat`.
The paper discloses `W_k = round(s*T_hat)` but does not define whether this
quantity is a radius or a full span. The independent implementation marks its
choice as inferred: it uses the symmetric radius
`max(1, round(s*T_hat/2))` around each of `t-T_hat` and `t+T_hat`. This avoids
collapsing all three scales at the shortest allowed period. When the
estimator's spectral confidence is exactly zero, the cycle positives are
omitted but the local positives remain.

For scale `s`, let `P_s(t)` be those positives and `A_s(t)` contain all
admissible candidates. The implemented multi-positive loss is:

```text
L_s(t) =
    log Σ[j in A_s(t)] exp(sim(z[t], z[j]) / τ)
  - (1 / |P_s(t)|) Σ[p in P_s(t)] sim(z[t], z[p]) / τ

L_PAMS = mean_s mean_valid_t L_s(t),       τ = 0.1
```

The denominator includes other valid times from the same video and every
valid frame from every other video in the physical batch, as stated by the
paper. Every five epochs, valid-frame mean embeddings for the full training
set are frozen and clustered with deterministic KMeans (`k=8`). For each
anchor, the explicit cross-cluster pool contains `positive_count` distinct
highest-similarity bank prototypes from other clusters, excluding every
current-batch video. This prototype-bank closure remains inferred because the
paper does not specify how cross-cluster samples are stored. Requested,
actual, and shortfall counts are logged; formal runs fail on any shortfall.

V14 also applies one target-free skeleton augmentation per encoder
optimization sequence: a centered three-axis rotation, isotropic scale, and
valid-frame Gaussian jitter. Period evidence, prototype refresh, SSHead
training, and inference retain the clean view. Rotation `+/-15` degrees and
scale `[0.85, 1.15]` reuse the supplement's robustness ranges; jitter
standard deviation `0.01` is independently inferred because the training
augmentation magnitudes are not published.

## Period Head audit and variants

The disclosed `PeriodHead` is `Linear(512,128) → GELU → Linear(128,1)`.
The paper applies its only stated training objective to `Z`, while saying
the head is exclusively used at inference. There is therefore no disclosed
way for `φ` to train.

- `PAMS-Literal` initializes the head with the experiment seed and never
  updates it. This is the executable consequence of the text.
- `PAMS-SSHead` freezes `Eθ` and trains `Hφ` for 30 epochs without counts:

```text
L_head = L_cycle + L_spectral + 0.1 L_variance + 0.01 L_smooth
```

`L_cycle` aligns values one estimated period apart; `L_spectral` maximizes
the non-DC power share in the fundamental bin and its immediate neighbours;
`L_variance` penalizes standard deviation below one; `L_smooth` penalizes
the squared second temporal difference. This is an inferred repair.

The separately fingerprinted
`sshead.input_source: projected_pose_reference_relative` route is a second,
more identifiable inferred repair. It never uses counts or action labels. For
each video it takes the same detached target-free period used during head
training, selects a dynamic phase anchor by maximizing full-period similarity
minus half-period similarity, and averages valid projected-pose rows at that
phase into a reference prototype `r`. Its head input and scalar teacher are:

```text
R[t,d] = zscore_valid((projected_pose[t,d] - r[d])^2)
y_ref[t] = zscore_valid(-mean_d (projected_pose[t,d] - r[d])^2)
P[t] = zscore_valid(H_phi(R[t]))

L_reference_relative = L_reference + L_fundamental + L_lag
                     + 0.1 L_low_frequency + 0.01 L_smooth
```

The negative distance makes each recurrence of the reference pose a maximum,
matching the downstream peak counter. `L_reference` regresses the centered
head stream to `y_ref`; unlike the legacy variance hinge, it has non-zero
gradient at a constant output. `L_fundamental` maximizes exact target-bin
power, `L_lag` aligns valid samples one period apart, and
`L_low_frequency` penalizes non-DC power below the target fundamental. The
four existing configured weights retain their stored fields but are
interpreted as lag, fundamental, low-frequency, and smoothness weights only
for this opt-in route; reference regression has unit weight. Zero-confidence,
insufficient-cycle, and constant-reference samples produce no fabricated
teacher. Training and inference call the same reference builder; inference
also applies the same continuous masked z-score to the raw head output before
the unchanged direct-FFT decoder and multi-expert counter consume `P`.

Training monitors this inferred repair without changing its loss. Each epoch
logs per-video valid-frame stream-standard-deviation summaries and collapse
fractions (`std <= 1e-6` and `std <= 1e-3`), together with maximum head
gradient RMS, maximum gradient-to-parameter L2 ratio, and zero-gradient step
count. Streams, loss components, and accumulated gradients must be finite.
Immediately before each optimizer step, training aborts if a loss above
`0.05` has gradient L2 at most `1e-12` while any valid-frame stream standard
deviation is at most `1e-6`, or if gradient RMS exceeds `10` or the
gradient-to-parameter L2 ratio exceeds `100`. A failed epoch produces neither
a checkpoint update nor a progress-log row. These preregistered, label-free
guards do not inspect benchmark counts.

## Multi-expert inference

The raw stream `P` supplies a fresh FFT period `T_hat`. In the historical
default `period.direct_fft_timebase: compact_valid` closure, the fallback
reference count is `floor(valid_frames/T_hat)`. Three experts use:

| Expert | Gaussian sigma | minimum peak distance |
|---|---:|---:|
| Fast | `0.05 T_hat` | `0.5 T_hat` |
| Medium | `0.12 T_hat` | `0.8 T_hat` |
| Slow | `0.15 T_hat` | `1.2 T_hat` |

The default inference FFT is applied directly to the compacted valid samples
of `P`, as ordered in Algorithm 1. The independent executable closure removes
an affine trend, applies one non-periodic Hann window, suppresses DC, and
searches only the configured period band. It does not FFT an autocorrelation
of `P`; that legacy decoder is retained only in historical artifact identities.

`period.direct_fft_timebase: dense_resampled` is an opt-in, independently
inferred final-readout missing-data closure. It preserves the complete
resampled clip clock
`0..L-1`: the affine mean and slope are fitted only at valid samples at their
dense indices, invalid samples contribute zero after detrending, and the Hann
window plus direct FFT operate over all `L` timeline positions. The evidence
gate still depends on the number of valid samples, while the frequency grid
and decoded period use `L`, and the fallback reference becomes
`floor(L/T_hat)`. This option is inference-only: it does not change period
estimation for encoder or Period Head training. In the reference-relative
path it also does not change the inference-time bootstrap period that builds
the head input; that upstream estimator retains its existing dense lag grid
and valid-count upper bound. Sparse masks can therefore truncate the bootstrap
period before this final readout runs, which remains a known, coverage-gated
limitation rather than a silently expanded fix. It also leaves the three
run-wise peak experts and the disclosed majority rule unchanged, and a fully
invalid clip still returns zero. This behavior is not presented as an author
setting because the paper does not specify how missing pose frames map onto
the inference FFT clock.

The dense fallback and peak experts intentionally have different observation
support: `floor(L/T_hat)` extrapolates a label-free cycle reference across
pose-missing intervals, while experts only accept peaks inside contiguous
valid runs. Majority agreement still wins before that fallback is consulted.
This support asymmetry is part of the frozen inferred closure and must pass
the synthetic-gap and train-only robustness gates; it is not evidence that
unobserved peaks were reconstructed.

Each computes short and long rolling statistics over `0.5 T_hat` and
`2 T_hat`. Its pointwise height threshold is:

```text
0.6 (μ_long + 0.6 σ_long) + 0.4 (μ_short + 0.6 σ_short)
```

Peak prominence must be at least `0.25(max(P_smooth)-min(P_smooth))`.
Missing frames divide the stream into contiguous valid runs; smoothing,
thresholding and peak finding are performed independently within each run, so
an invalid gap can never synthesize or alter a cross-gap peak.
Two agreeing experts form the majority. Otherwise the candidate nearest the
FFT reference is selected; ties prefer Medium, then Fast, then Slow.
The public confidence is the vote confidence multiplied by the FFT period
confidence; generic callers that do not supply a period confidence retain the
neutral multiplier of one.
Per-expert peaks, thresholds and counts are retained for audit, while the
public `CountResult` contains the compact final result.

## Optimization and checkpoint state

Encoder training uses AdamW with learning rate and weight decay `1e-4`,
effective batch 32, 150 epochs and ReduceLROnPlateau (`factor=.5`,
`patience=8`, minimum learning rate `1e-6`). A resumable checkpoint contains
model, optimizer, scheduler, epoch, cluster labels, random seed and
configuration fingerprint.

The training API receives only `PoseSequence` objects. Counts enter only the
separate evaluator, which uses the shared NMAE/OBO implementation.
