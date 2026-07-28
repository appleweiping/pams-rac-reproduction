# Local synthetic safety evidence

These files are deterministic, label-free engineering diagnostics generated
from the reviewed pre-publication worktree. They are not UCFRep results and
cannot satisfy the repository's benchmark verification gate. The same
commands are rerun in the committed server container before training.

## Counter sign/phase gate

`counter-sign-phase-local.json` covers all 576 frozen combinations:

- counts `2,5,10,20,30,40`;
- 16 phases and both stream signs;
- sinusoid, second-harmonic, and circular-Gaussian waveforms.

This is a component-isolation test. Each case supplies the exact synthetic
period `256 / count` to the multi-expert counter; it does not run the learned
encoder, Period Head, or FFT period estimator and cannot substantiate
end-to-end PAMS correctness.

Result: **pass**. NMAE `0.0642650463`, OBO `1.0`, maximum absolute error
`1`, and sign-pair signed bias `0.2048611111`. The mean signed prediction
error `-0.453125` remains visible as a non-gating diagnostic.

SHA-256:
`d564ef104b99655aad00d1e80f3a7531675cc627e8cf7e128345ffa2c605475c`.

## Period-estimator plus counter gate

`period-counter-sign-phase-local.json` runs the same 576 streams through the
mask-aware FFT/autocorrelation period estimator and then the multi-expert
counter. It supplies no exact period to the code under test, but still
bypasses the learned encoder and Period Head.

Result: **pass**. All 576 dominant periods are recovered exactly at FFT-bin
resolution (`max_period_relative_error=0`), minimum spectral confidence is
`0.4380851220`, NMAE is `0.0642650463`, OBO is `1.0`, and maximum absolute
count error is `1`.

SHA-256:
`e926b963327ea1c41827ecd3d7ae28ef5bd0d1e2d416b16771defe55fe51f4eb`.

## SSHead collapse diagnostic

`sshead-collapse-local.json` covers 160 combinations of four constants, four
periods, full/fixed-gap masks, and four near-constant amplitudes.

Result: **fail, retained intentionally**. All 32 exact-constant cases are
positive-loss zero-gradient dead points. Of 128 near-constant cases, 32
exceed the frozen gradient-RMS limit of 10; the maximum is
`2080.0947552`. Every reported value is finite.

SHA-256:
`b7f54f6d7bc643d033a2a5758ceb43a5500d8c76a9d4898f79af2431973cc3aa`.

The inferred loss is not silently changed to hide this result. Runtime
SSHead training instead measures the observed valid-stream dispersion and
head gradients, and aborts before `optimizer.step()` if exact collapse or a
near-collapse gradient spike occurs.
