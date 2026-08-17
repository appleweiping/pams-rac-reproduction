# TempoRAC Contract Amendment 001: Executable X0 Orbit Scale

> **STATUS: `PROPOSED_PENDING_FRESH_REVIEW`**
>
> **SAME-FAMILY PROVISIONAL — NON-AUTHORITATIVE**
>
> **NO GATE PASSAGE, TRAINING, DATA ACCESS, RESULT, CLAIM, PAPER PROMOTION, SERVER LAUNCH, OR GIT AUTHORIZATION**

**Date:** 2026-08-16

**Would amend, only if accepted:** `refine-logs/temporac/FINAL_PROPOSAL.md` Section 6.1, formula F8 and its immediately following RMS definition

**Scope:** one dimensionless X0 displacement scale and an explicit RMS domain; all other TempoRAC method, data, topology, target, training, evaluation, resource, and claim contracts remain unchanged

## 1. Discovered contradiction

The frozen F8 contract defines `w=H^T rho/||H^T rho||` and constructs the moving 16-coordinate displacement as

```text
0.12*w*sin(2*pi*phi)
+ sum_(h=2)^4 (0.03/h)*(u_h*cos(2*pi*h*phi)+v_h*sin(2*pi*h*phi))
```

where the two-pass modified Gram--Schmidt procedure makes `w,u_2,v_2,u_3,v_3,u_4,v_4` orthonormal. The same section rejects a source attempt whenever, for any `d=2,...,8`, the RMS between `G(phi)` and `G(phi+1/d)` is at most `0.05`. RMS is defined as the square root of summed squared coordinate differences divided by coordinate count.

Under the only contract-consistent post-normalization interpretation, the full-grid RMS over the 34 COCO-17 `x,y` coordinates is independent of source ID and attempt and equals

```text
D_1(d) = sqrt(
  [0.12^2*(1-cos(2*pi/d))
   + sum_(h=2)^4 2*(0.03/h)^2*(1-cos(2*pi*h/d))] / 34
)
```

For `d=2,...,8`, respectively,

```text
0.0293056911, 0.0256926117, 0.0213514017, 0.0181527440,
0.0157589844, 0.0138966078, 0.0124074026
```

Every value is below `0.05`, so all 64 attempts for every source must be rejected. Dividing by only the 16 moving coordinates also fails, with values ending at `0.0180867419`. Using raw, unnormalized `H^T rho` would pass but directly contradicts the explicit unit-L2 normalization and leaves the MGS basis semantics inconsistent. No allowed interpretation makes the frozen X0 executable.

The existing production implementation therefore correctly fails closed. This amendment does not reinterpret that failure as a pass.

## 2. Sole normative correction

If and only if this amendment passes a fresh review, insert the exact dimensionless constant

```text
s_X0 = 25/6
```

and replace F8 only by

```text
G_n(phi)=B+E_J[s_X0 * (
  0.12*w*sin(2*pi*phi)
  + sum_(h=2)^4 (0.03/h)*(u_h*cos(2*pi*h*phi)+v_h*sin(2*pi*h*phi))
)]
```

Equivalently, the fundamental amplitude becomes exactly `0.5` and each paired harmonic amplitude becomes exactly `0.125/h`. The scale multiplies every periodic displacement term, not the base pose, confidence, clock, masks, phase, target pulse, responsibility, static code, duration, offset, or any natural input.

The binary64 evaluation order is normative. Production first computes the
existing unscaled bracket exactly as written: assign the fundamental term,
then add the `h=2`, `h=3`, and `h=4` paired terms in increasing order using
the existing in-place accumulation. Only after that complete bracket exists,
execute one outer multiplication by `np.float64(25.0/6.0)`. The analytic
tangent follows the identical rule: finish the existing fundamental and
increasing-harmonic accumulation, then multiply the complete tangent once by
the same scalar. Distributing the scalar into individual terms, scaling basis
vectors, or changing the MGS inputs is forbidden because those alternatives
can change committed last bits even when their real-number formulas agree.

For the separation and reversal predicates in Section 6.1, freeze RMS to mean the binary64 RMS over the complete exact `4096 x 34` array of COCO-17 `x,y` coordinate differences. The phase grid remains `r/4096`, `r=0,...,4095`; the 18 fixed coordinates remain in the denominator as exact zero differences. This makes explicit the domain already used by `G`, without changing either threshold.

For each `d=2,...,8`, production must retain this exact reduction sequence:
evaluate `pose=G(phi)` and `shifted=G((phi+1/d) mod 1)` as C-contiguous
binary64 arrays; form `difference=pose-shifted`; call
`np.square(difference)`; call
`np.mean(...,dtype=np.float64)` over all `4096*34` C-order elements; cast the
result once to Python `float`; then call `math.sqrt`. The pinned NumPy 2.1.0
environment is part of the S0 byte commitment. The reversal predicate retains
the proposal implementation's literal C-order flatten, reversed-grid indexing,
`np.fft.rfft`/conjugate-product/`np.fft.irfft`, energy, nonnegative clamp, and
minimum sequence; the closed-form reversal value below is an audit witness,
not a replacement implementation.

Nothing else changes. In particular:

- `w` remains unit L2 and the seven MGS directions remain orthonormal
- all rejection thresholds remain `0.05`, `1e-3`, and `0.01` as written
- the 40 source IDs, splits, PCG64 seeds, attempts `0..63`, draw order, and first-passing-attempt rule remain unchanged
- F9--F23, the landmark score, phase, ten pulses, masks, ownership, topology bank, resamplers, teacher, certificate, experts, router, NOLA, decoder, losses, jobs, seeds, checkpoints, gates, receipts, metrics, budget, protocol labels, and evidence ceiling remain unchanged
- no natural-data, evaluator, label, server, paper, or public-claim authority is created

## 3. Independent executable evidence required for acceptance

The current implementation shard contains a read-only mathematical amendment probe in `tests/temporac/test_x0.py`; production `src/pams/temporac/x0.py` still uses the unamended scale and fails closed. Because that pre-acceptance probe distributes the common scale across an orthonormal coefficient basis, it proves the guards and decisions but is not a byte-level orbit-hash witness. Fresh review must additionally replay the exact outer-multiplication and literal FFT operation order frozen above over all 40 source IDs and all 64 attempts. The mathematical probe exhaustively applies the frozen PCG64 seed and MGS construction, evaluates the exact 4096-point grid, and obtains:

| Check | Observed value under `s_X0=25/6` | Frozen requirement |
|---|---:|---:|
| MGS candidates | `2560/2560` finite and orthonormal | every attempted basis valid or explicitly rejected |
| Retained source orbits | `40/40`, all at attempt `0` | exactly 40 rows; first pass retained |
| Coordinate range | `[-1.0615805372092377, 0.95]` | within `[-2,2]` |
| Pose scale | `0.65` | above `0.1` |
| 66-geometry grid arc | `[4.088598312896032, 4.317988354684301]` | above `1` |
| Analytic tangent norm | `[1.3603495231756633, 3.423471224691923]` | above `1e-6` |
| Separation, `d=2,...,8` | `0.1221070461, 0.1070525486, 0.0889641736, 0.0756364334, 0.0656624349, 0.0579025323, 0.0516975107` | each strictly above `0.05` |
| Circular state-collision lower bound | `0.014291548761875736` | strictly above `1e-3` at separated pairs |
| Reversal RMS minimum | `0.01973191444620664` at `c=1/2` | strictly above `0.01` |
| Landmark predicate | one strict maximum, one strict minimum, one strict high-score component | exact predicate |

The analytic coordinate excursion bound is `0.7708333334`, so combining it with the frozen base-pose extrema also proves the `[-2,2]` guard before the finite scan. The separation values obey `D_s(d)=s_X0*D_1(d)` exactly because the seven displacement directions are orthonormal. The new scale does not move phase landmarks or pulse locations because it is strictly positive and common to all periodic terms.

A fresh reviewer must independently rederive the formula, rehash the bound inputs, and rerun or independently reproduce all 2,560 candidate checks. A review may not accept the amendment by relying only on this document's table.

## 4. Non-expansion and failure behavior

This is a numerical executability correction, not a new mechanism, dataset, loss, baseline, ablation, seed, job, metric, claim, or contribution. It does not repair a future failed gate, relax a threshold, remove a source, select a favorable attempt after results, or authorize training. It preserves the experiment inventory at one primary claim, four blocks, three baseline families, 27 training jobs, 93 allocated A6000-hours, maximum concurrency two, and the 64 GiB artifact cap.

Until a fresh review accepts the amendment and the canonical full proposal is patched and rehashed:

- production X0 must continue to use the original F8 bytes and fail closed
- P00 may test and report the contradiction but cannot issue a passing X0 receipt
- P01/P2 may exercise unaffected synthetic interfaces but cannot freeze S0
- no G0, K0, teacher job, response job, natural-data acquisition, evaluator capability, or server launch may occur
- no result, abstract number, title claim, main-paper statement, or Git release may cite the amended scale as canonical

If the amendment is rejected, the TempoRAC execution route remains blocked at X0 and must return to research refinement; no alternate constant may be searched after seeing efficacy results.

## 5. Bound snapshot

| Input | SHA-256 |
|---|---|
| `refine-logs/temporac/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` |
| `refine-logs/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` |
| `refine-logs/EXPERIMENT_PLAN.md` | `3fd1986871b32674d74b395fbfc9720712d9fba0c8583c24c1d5bddb87086eb6` |
| `refine-logs/EXPERIMENT_TRACKER.md` | `c7e2f32c3c81b66884abb13760a671ecc7cdd314fd6c994a4ed4c46cb754a63b` |
| `src/pams/temporac/x0.py` | `1690abb9bc6b6da4a25a4467162c8ddc111fe0a3c8cf33e7f70f0faa8e84f6f4` |
| `tests/temporac/test_x0.py` | `a2e13ad508777d76a0e31075b4ea7dcff843e6f4168096736b98f008ade48620` |
| `idea-stage/docs/research_contract.md` | `b04e5bf5606ad8be1bcc8000098835ab525ba054c1f33d565ad72093e7c47eca` |

The amendment remains `PROPOSED_PENDING_FRESH_REVIEW`, same-family provisional, non-authoritative, and non-executing.
