# PAMS-SSHead train337 loss/gradient diagnosis

**Status:** root cause localized; reproduction still partial

**Scope:** UCFRep-526 train337 poses only; no counts, dev inputs/targets, or
test105 access

**Diagnosed source:** `ebbc296db11c3d7d06f5fc3090a3487abed55e36`

The historical SSHead did receive gradients and did train. Its catastrophic
readout is not caused by a missing `backward()`, an accidental detach of the
head, or a missing optimizer step. The primary failure is instead a
positional shortcut: the pointwise head consumed post-position-encoding
Transformer embeddings, and the independently inferred phase-free objective
made a nearly video-invariant absolute-position waveform a low-loss solution.

There is also a separate loss defect. An exactly constant stream has positive
total loss `1.1000000238`, but the gradients of cycle, spectral, variance,
smoothness, and total loss are all exactly zero. Thus the advertised variance
hinge is not a mathematical safeguard against exact collapse.

The machine-readable evidence, full precision values, checkpoint bindings,
and claim boundary are in
[`pams_sshead_gradient_diagnosis_train337.json`](pams_sshead_gradient_diagnosis_train337.json).
The read-only reproducer is
[`scripts/diagnose_sshead_gradients.py`](../../scripts/diagnose_sshead_gradients.py).

## What was ruled out

The training path freezes the encoder and detaches its output, but the Period
Head remains in the graph. Across all six final logs—three adaptive
multi-scale seeds and three fixed-period-16 seeds—`zero_grad_steps=0`.
The two representative trained heads moved by `10.3%` and `11.7%` of their
initial parameter L2 norms.

The heads also learned the supplied pseudo-target:

| train337 probe | Target period | Final emitted period | Parameter change |
|---|---:|---:|---:|
| Fixed-period-16, seed 2026 | 16.0 | median 15.06 | 11.7% |
| Adaptive multi-scale, seed 2026 | median 7.76 | 8 on 310/337 videos | 10.3% |

This is positive evidence that gradients, `backward()`, and AdamW steps were
working. It is negative evidence for the objective: the adaptive head learned
the pseudo-period supplied by the PE-contaminated representation.

## The positional-template mechanism

Among the 155 fully valid train videos, pairwise correlation of centered
final Period Head streams is:

| Probe | Mean correlation | Median correlation | Mean stream std |
|---|---:|---:|---:|
| Fixed-period-16 | 0.99085 | 0.99191 | 0.002251 |
| Adaptive multi-scale | 0.99871 | 0.99884 | 0.007711 |

The adaptive random head already had median correlation `0.99792`; training
increased it to `0.99884`. A pointwise `512→128→1` head has no temporal
mechanism that could force this shared waveform to follow pose phase instead
of absolute frame index.

This train-only result independently supports the prior read-only dev
artifact, where all fully valid videos produced the same period-8 waveform.
The present diagnostic did not open or rescore that development data.

## Four losses and their gradients

Values below are from the final seed-2026 train337 checkpoints. Gradient L2
is measured after each configured weight is applied.

| Probe | Component | Loss | Effective weight | Parameter-gradient L2 |
|---|---|---:|---:|---:|
| Fixed T=16 | Cycle | 0.00000109 | 1.0 | 0.0000211 |
| Fixed T=16 | Spectral | 0.166069 | 1.0 | 1.22313 |
| Fixed T=16 | Variance | 0.997749 | 0.1 | 0.001138 |
| Fixed T=16 | Smoothness | 0.000000587 | 0.01 | ≈0.000000102 |
| Adaptive | Cycle | 0.00000824 | 1.0 | 0.0000983 |
| Adaptive | Spectral | 0.247344 | 1.0 | 0.226926 |
| Adaptive | Variance | 0.992288 | 0.1 | 0.004085 |
| Adaptive | Smoothness | 0.0000258 | 0.01 | ≈0.00000276 |

The weighted spectral gradient is `1074.6×` the weighted variance gradient
for fixed-period training and `55.6×` for adaptive training. A naive
`variance_weight: 0.1 → 1.0` therefore cannot be expected to balance either
case, much less both.

The scale sweep also shows that the soft hinge does not enforce unit standard
deviation. Scaling the fixed-period output by `128×` raises its mean std only
to about `0.288`, yet reduces total loss from `0.265845` to `0.255263`.
For the adaptive output, the best sampled region is around `32×`, or mean std
about `0.247`. These are not optima at the nominal margin `std=1`.

Amplitude collapse is nevertheless not the direct reason for the giant count:
the downstream peak rules are largely affine-scale invariant. The count
multiplier comes from the shared high-frequency positional waveform.

## Exact stationary point and invalid-row bug

For a zero stream of shape `[1,256]`, period 16, and all frames valid:

| Component | Loss | Stream-gradient L2 | Nonzero gradient entries |
|---|---:|---:|---:|
| Cycle | 0 | 0 | 0 |
| Spectral | 1 | 0 | 0 |
| Variance | 1 | 0 | 0 |
| Smoothness | 0 | 0 | 0 |
| Total | 1.1 | 0 | 0 |

`relu(1-std)` cannot escape exact `std=0`, because the standard-deviation
gradient is zero there. The clamped zero-spectrum ratio also supplies no
gradient. This is an objective defect independent of the positional shortcut.

A smaller implementation bug affects eight train rows with fewer than two
valid frames:

- they add variance loss `1` with no gradient;
- the collapse logger maps them to `std=0`;
- `stream_std_min` is therefore always zero, and
  `collapsed_fraction_1e6 = 8/337 = 0.0237388724` for every epoch.

Those rows should be excluded from the variance average and reported as
unusable data, not as collapsed model outputs. This dilutes the useful
variance gradient by only `2.37%`, so it is not the catastrophic root cause.

## Strict conclusion and next experiment

The evidence supports the explicit pre-PE SSHead-input repair and does not
support a variance-only repair. The clean next experiment is to run the
pre-PE variant alone as a one-variable causal test. A variance-weight or loss
scan should run only on synthetic inputs and label-free train337 diagnostics,
not be selected with development targets while the input-source experiment is
in flight.

If pre-PE input is insufficient, the next loss candidate should:

1. skip rows with fewer than two valid frames;
2. have a non-vanishing scale-restoring gradient near collapse, for example a
   collapse-safe log-ratio hinge or a constrained/augmented-Lagrangian
   variance condition;
3. pass exact-constant, finite-gradient, period-concentration,
   cross-video-stream-correlation, and invalid-row gates before any
   development scoring.

No training semantics were changed by this diagnosis, and it is not a
verified reproduction or a sealed-test result.
