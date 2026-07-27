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

The paper defines the general input as `K×3`; `K=33`, 256-frame resampling,
MediaPipe extraction and per-frame min-max normalization are independent
closures. Invalid frames are all zero and are masked in attention, loss,
period estimation and inference.

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

## Stop-gradient period estimate

The paper says the period is estimated by applying FFT to an autocorrelation
of stop-gradient embeddings or a pose-energy proxy. The implementation:

1. selects the highest-variance signed pose/embedding coordinate as a
   phase-preserving scalar component;
2. mean-centres it over valid frames;
3. computes mask-normalized, non-circular autocorrelation with FFT;
4. applies a Hann window and finds dominant spectral power corresponding to
   periods from 4 through 128 frames.

The first ten encoder epochs use pose coordinates. Later epochs use detached
encoder embeddings. Period selection has no gradient.

## PAMS TCC

For each anchor `z[t]`, local positives are valid `z[t-1]` and `z[t+1]`.
Past/future cycle positives are the highest-cosine-similarity valid frames in
the frozen correspondence neighbourhood around `t-T_hat` and `t+T_hat`.
The window width is scaled by `s ∈ {0.5,1.0,1.5}` and the search radius is
10% of that width.

For scale `s`, let `P_s(t)` be those positives and `A_s(t)` contain all
admissible candidates. The implemented multi-positive loss is:

```text
L_s(t) =
    log Σ[j in A_s(t)] exp(sim(z[t], z[j]) / τ)
  - log Σ[p in P_s(t)] exp(sim(z[t], z[p]) / τ)

L_PAMS = mean_s mean_valid_t L_s(t),       τ = 0.1
```

The denominator includes other valid times from the same video and pooled
prototypes from other videos. Every five epochs, valid-frame mean embeddings
are clustered with deterministic KMeans (`k=8`). Up to the anchor's positive
count of highest-similarity different-cluster prototypes is added as the
explicit cross-cluster hard-negative pool.

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

## Multi-expert inference

The raw stream `P` supplies a fresh FFT period `T_hat` and reference count
`floor(valid_frames/T_hat)`. Three experts use:

| Expert | Gaussian sigma | minimum peak distance |
|---|---:|---:|
| Fast | `0.05 T_hat` | `0.5 T_hat` |
| Medium | `0.12 T_hat` | `0.8 T_hat` |
| Slow | `0.15 T_hat` | `1.2 T_hat` |

Each computes short and long rolling statistics over `0.5 T_hat` and
`2 T_hat`. Its pointwise height threshold is:

```text
0.6 (μ_long + 0.6 σ_long) + 0.4 (μ_short + 0.6 σ_short)
```

Peak prominence must be at least `0.25(max(P_smooth)-min(P_smooth))`.
Two agreeing experts form the majority. Otherwise the candidate nearest the
FFT reference is selected; ties prefer Medium, then Fast, then Slow.
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
