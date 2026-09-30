# SSHead next-candidate audit: train337 and synthetic only

**Status:** bounded candidate audit complete; reproduction remains partial

**Firewall:** 337 label-free training pose caches and deterministic synthetic
skeletons only. No training counts, development inputs/targets, or test105
were mounted or read.

The machine-readable evidence is in
[`pams_sshead_next_candidate_train337.json`](pams_sshead_next_candidate_train337.json).
The independent runner is
[`scripts/diagnose_sshead_phase_anchor.py`](../../scripts/diagnose_sshead_phase_anchor.py).

## Decision

Two proposed repairs are rejected:

1. A zero-pose contextual residual does not provide a clean contextual
   replacement for pre-PE features.
2. A signed pose-band phase loss has a valid anti-collapse gradient, but the
   disclosed pointwise head does not learn it on train337.

Continuous period-confidence weighting remains the only justified
single-variable formal experiment. It corrects a real semantic problem—the
old loss treated every nonzero confidence identically—but the train-only
probe does not itself show that it solves counting.

## Feature-source comparison

This is a random-head, read-only comparison on the same 337 training caches.

| Head input | Median cross-video correlation | Spectral loss | Median emitted period |
|---|---:|---:|---:|
| Post-PE encoder embedding | 0.8627 | 0.9137 | 6.56 |
| Pre-PE projected pose | 0.0015 | 0.8500 | 85.33 |
| Encoder minus zero-pose encoder | 0.5665 | 0.9020 | 36.57 |
| L2-normalized zero-pose residual | 0.5663 | 0.9012 | 36.57 |

Pre-PE projection removes the shared positional waveform. The zero-pose
residual still has strong cross-video correlation and worse target-band
spectral loss, so contextual subtraction is not the next experiment.

Pre-PE also exposes the remaining problem: its random head emits period 128
for 114/337 videos even though the training-target median is 9.14.

## Why continuous confidence is justified

Mean period confidence is only `0.147`; the positive range is approximately
`0.044–0.911`. The historical nonzero gate gives both endpoints equal weight.
Normalized continuous weighting reduces the effective training sample size to
`209.9`, emphasizing the better-supported pseudo-periods without consulting a
label.

This is a principled loss-semantics correction, but not yet evidence of
accuracy. After 30 epochs, the train-only confidence-weighted head still has:

- median emitted period `85.33`;
- period-128 fraction `35.0%`;
- confidence-weighted target-period relative error `5.99`;
- only `15.4%` of positive-confidence rows within 10% of their training
  target.

## Phase-anchor experiment

For each video, the phase teacher reconstructs projected pose from the target
FFT bin and its two neighbors, extracts the dominant signed temporal
component, and standardizes it over valid frames. The phase loss is
continuous-confidence-weighted centered MSE.

Unlike the current variance hinge, this term does have a gradient at exact
collapse: a constant 256-frame stream produces loss `1`, gradient L2 `0.125`,
and nonzero gradients on all 256 entries.

That mathematical property does not make the teacher learnable by the
pointwise head:

| Phase weight | Phase loss, epoch 1 → 30 | Mean phase correlation | Weighted target-period error | Period-128 fraction | Mean stream std |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.0016 → 1.0021 | -0.029 | 5.99 | 35.0% | 0.0138 |
| 0.25 | 1.0015 → 1.0016 | -0.033 | 6.24 | 35.3% | 0.0126 |
| 1.0 | 1.0013 → 1.0005 | +0.018 | 6.45 | 35.6% | 0.0108 |

The phase term stays at the chance-level MSE of about one, worsens target
period agreement, and reduces rather than restores stream amplitude. Weight
tuning is therefore stopped.

## Synthetic interpretation

All three candidates recover the expected period within 10% for all 39
synthetic counts from 2 through 40. Their downstream count is nevertheless
always one low: exact rate `0`, OBO `1`, NMAE `0.0841`. This is the finite
sequence endpoint convention—the peak counter sees one fewer interior peak—
and is identical across candidates.

Consequently, synthetic period recovery remains a useful gate; synthetic
exact count cannot distinguish these heads until the endpoint rule is
explicitly corrected and refrozen.

## Next implementation only if formal confidence weighting fails

Use an explicitly named inferred architecture, not another loss-weight scan.
Prepend one depthwise temporal operator and keep the disclosed MLP intact:

`projected_pose_pre_pe → depthwise Conv1d(512,512,k=5,groups=512) →
Linear(512,128) → GELU → Linear(128,1)`.

Initialize each depthwise kernel to the identity delta `[0,0,1,0,0]`. This
adds only `2,560` parameters, starts exactly from the current pre-PE head
input, and learns a short PE-free temporal receptive field. Inputs and outputs
outside valid frames remain zero. The pointwise MLP sees pose configuration
but cannot directly resolve local direction or velocity; the single
depthwise operator can, without reintroducing absolute position encoding.

The first temporal-head experiment should keep continuous confidence
weighting and omit the rejected phase term so only architecture changes. It
must pass these frozen, label-free gates before any development scoring:

- confidence-weighted target-period relative error `≤0.25`, with at least
  `80%` within 10%;
- period-4 and period-128 fractions each `≤10%`, and no rounded-period mode
  above `25%`;
- absolute median fully-valid cross-video correlation `≤0.1`;
- usable median stream std `≥0.1`, with at most `5%` near zero; unusable rows
  excluded;
- 100% synthetic 2–40 period recovery within 10%, variable-speed error
  `≤10%`, and pause error `≤15%`;
- finite gradients and zero zero-gradient optimizer steps.

These thresholds are frozen from synthetic semantics and the nominal SSHead
variance margin, not development labels. Failing them stops the candidate.

The successful server report is bound by SHA-256
`ebc843561566b302caa14437cf9b6a47980d9c002c0903d4ebc458e048addb01`.
An initial launch failed before loading any input because the image did not
contain `scripts/`; it is retained as an excluded startup failure.
