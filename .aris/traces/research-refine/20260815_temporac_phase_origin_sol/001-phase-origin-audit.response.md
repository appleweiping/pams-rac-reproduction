## Bottom-line ruling

The proposed repair is conditionally valid, but its claim must be narrowed.

A weighted circular origin removes a constant phase-rotation gauge exactly if its weights depend only on a fixed physical-state landmark and the circular resultant is nonzero. It is not automatically invariant to monotone resampling merely because the state is “clock-blind”: uniform sample weights, velocity magnitudes, and fixed-frame delay lags all change under resampling. Exact invariance requires traversal-measure weighting and a warp-invariant projection/state representation.

The repair does not identify a human repetition convention. It imposes a deterministic pose-defined seam. It also removes only the constant \(S^1\)-rotation gauge, not the larger orientation-preserving reparameterization gauge admitted by a reconstruction-based degree-one teacher. Degree \(+1\) is sufficient for one seam crossing per complete observable orbit traversal, but not for deciding whether humans group two observable primitive traversals as one repetition, where a repetition starts, or how partial repetitions are counted.

## 1. Conditional theorem and assumptions

Let \(\gamma:S^1\to\mathcal Y\) be the masked delay-state trajectory of one action, and let \(z=f(\gamma(\phi),c)\in S^1\).

The desired result holds under all of the following assumptions:

1. **Primitive embedded orbit.** \(\gamma\) is an embedding of \(S^1\), with trivial rotational stabilizer. The human repetition unit is assumed, rather than learned, to equal one forward traversal of this orbit.
2. **Teacher topology.** For fixed content \(c\), \(f|_{\gamma(S^1)}\) is an orientation-preserving homeomorphism of degree \(+1\). The decoder has no time/content bypass.
3. **Cross-view stability.** Changing clip sampling, repetition count, or static code may change \(z\) only by a constant rotation:
   \[
   z'(y)=e^{i\beta}z(y).
   \]
   If the static code changes the phase by a nonlinear homeomorphism, origin rotation alone does not align the targets.
4. **Projection validity.** The landmark score \(s=p(y,m)\) is fixed independently of the teacher, invariant at corresponding physical states under every allowed warp, and valid on a fixed mask support.
5. **Unique landmark.** \(p\circ\gamma\) has one isolated global maximum per primitive orbit. Repetitions of that maximum across cycles are permitted; two separated maxima within one cycle are not.
6. **Complete support.** At least one uninterrupted complete orbit traversal is observed. A partial clip cannot certify that its observed maximum is the global orbit maximum.
7. **Sampling/no-alias assumption.** Every valid adjacent sample interval contains strictly less than \(1/4\) of a true traversal, and cannot hide an integer number of complete cycles. This is an acquisition assumption; modulo-\(S^1\) phase cannot verify it from endpoints.
8. **Orientation.** Allowed warps are orientation-preserving. Pauses are permitted only if phase remains constant; reversal is outside the contract.
9. **Origin concentration.** The weighted circular resultant is separated from zero and passes the uniqueness/stability gates below.

For complete traversals \(I_j\), define a phase-traversal-measure circular mean

\[
c_j=
\frac{
\int_{I_j}
e^{\kappa(s(t)-s_j^{\max})}z(t)\,dA(t)}
{
\int_{I_j}
e^{\kappa(s(t)-s_j^{\max})}\,dA(t)
},
\qquad
C=\frac1K\sum_{j=1}^Kc_j,
\qquad
o=\frac C{|C|},
\]

where \(A\) is the unwrapped positive teacher phase in cycles. Set

\[
\widetilde z(t)=z(t)\overline{o}.
\]

Then:

- For \(z'=e^{i\beta}z\), \(C'=e^{i\beta}C\), \(o'=e^{i\beta}o\), and \(\widetilde z'=\widetilde z\). Thus constant phase rotation is removed exactly.
- Under an orientation-preserving time change, the Riemann–Stieltjes integral with respect to \(dA\) is unchanged. Uniform sample averaging does not have this property.
- Repeating an identical complete cycle \(K\) times leaves \(o\) unchanged; each cycle contributes equal normalized mass.
- On every complete positive traversal, half-open integer crossings of the canonical unwrapped phase occur exactly once.
- If the final NOLA response has one connected supra-\(0.5\) component per target event, separated by sub-\(0.5\) valleys and no extra supra-threshold support, one connected-component decoder returns exactly one peak per event.

Important limitation: replacing \(z\) by a general orientation-preserving homeomorphism \(H(z)\) need not carry the finite-\(\kappa\) circular mean to \(H(o)\). The full phase-parameterization gauge remains. As \(\kappa\to\infty\) with a unique landmark, the origin approaches the teacher phase of that landmark, but finite concentration requires an explicit landmark-alignment gate.

## 2. Exact stable algorithm

### Preprocessing

1. Collapse adjacent equal source clocks before constructing delay states.
2. Use a deterministic mask/confidence-weighted coordinate average.
3. If duplicate-clock observations disagree by more than the frozen pose tolerance, or the same clock reappears non-contiguously, abstain.
4. Form maximal valid temporal segments. Never unwrap, generate pulses, or join decoder components across an invalid gap.

### Fixed projection

With frozen \(a,\mu,\sigma\), compute in float64

\[
s_t=
\frac{\sum_d m_{td}a_d(y_{td}-\mu_d)/\sigma_d}
{\sqrt{\sum_d m_{td}a_d^2}+\epsilon_p}.
\]

Exact invariance requires the nonzero support of \(a\) to be observed and warp-invariant. Mask normalization alone does not make two different masks equivalent. In particular, raw velocity magnitudes and fixed-frame lags must not enter this projection unless a synthetic correspondence test proves their invariance.

### Teacher phase and traversal blocks

Normalize every teacher output:

\[
z_t\leftarrow z_t/\max(\|z_t\|_2,\epsilon_z).
\]

For each valid edge,

\[
\delta_t=\frac1{2\pi}
\operatorname{atan2}
\left(
z_t^{(1)}z_{t+1}^{(2)}-z_t^{(2)}z_{t+1}^{(1)},
\langle z_t,z_{t+1}\rangle
\right).
\]

Accumulate \(\delta_t\) with compensated float64 summation, starting from zero in each valid segment. Partition the path into complete progress intervals \([j,j+1]\); this avoids using the arbitrary raw phase seam. Split an edge at a block boundary by deterministic interpolation. Exclude the trailing partial interval. If there is no complete interval, abstain.

Any \(\delta_t<-10^{-12}\) is not a numerical pause and triggers the orientation/backtracking audit. Any \(|\delta_t|\ge0.25\) fails the supported sampling domain. The \(0.25\) bound still cannot detect an unobserved whole cycle between identical endpoints; that remains an explicit acquisition assumption.

### Stable weighted circular mean

For each quadrature subinterval \(b\), let \(\lambda_b>0\) be its phase length, \(s_b\) its midpoint projection, and \(z_b\) the normalized spherical midpoint of its endpoint phases. Compute

\[
\ell_b=\log\lambda_b+\kappa(s_b-s_j^{\max}),
\qquad
L_j=\max_b\ell_b,
\]

\[
c_j=
\frac{\sum_b e^{\ell_b-L_j}z_b}
{\sum_b e^{\ell_b-L_j}}.
\]

Use pairwise or compensated summation. Then compute \(C=K^{-1}\sum_jc_j\), \(\rho=|C|\), and \(o=C/\rho\). This is stable against exponent overflow, sampling density, and repeated pause samples. Ordinary \(\sum_t w_tz_t\) is unacceptable.

Canonicalize using real-vector complex multiplication:

\[
\widetilde z_t=
\left(
z_{t,x}o_x+z_{t,y}o_y,\;
z_{t,y}o_x-z_{t,x}o_y
\right).
\]

### Half-open pulse generation

Let

\[
u_t=\operatorname{mod}\!\left(
\frac{\operatorname{atan2}(\widetilde z_{t,y},\widetilde z_{t,x})}{2\pi},1
\right).
\]

Initialize \(A_0=u_0\) and continue with \(A_{t+1}=A_t+\delta_t\), using compensated summation. For numerical ownership of seam endpoints, define

\[
S_\tau(x)=
\begin{cases}
\operatorname{round}(x),&
|x-\operatorname{round}(x)|\le\tau,\\
x,&\text{otherwise},
\end{cases}
\qquad \tau=10^{-7}\text{ cycles}.
\]

Then

\[
e_t=
\left\lfloor S_\tau(A_{t+1})\right\rfloor
-
\left\lfloor S_\tau(A_t)\right\rfloor.
\]

Accept only \(e_t\in\{0,1\}\). A negative result is a backward seam crossing; a value above one is aliasing. A pause exactly at the seam produces one pulse on entry and zero on all zero-increment pause edges.

The pulse array must also be unchanged for \(\tau\in\{10^{-8},10^{-7},10^{-6}\}\); otherwise the track is seam-numerically ambiguous and must abstain.

### NOLA and the single decoder

Use only valid, non-padding global edges:

\[
R_e=
\frac{\sum_{\ell:e\in W_\ell}a_{\ell e}r_{\ell e}}
{\sum_{\ell:e\in W_\ell}a_{\ell e}+10^{-8}}.
\]

Require a finite denominator at least \(10^{-3}\). Split at invalid gaps, threshold once at \(R_e\ge0.5\), and count connected components once per identity. For a plateau maximum, choose the lower-index midpoint of the maximal-value plateau. Do not add NMS, a minimum-distance rule, or an integral fallback.

## 3. Deterministic synthetic tests and thresholds

Freeze generator hashes and all constants before natural evaluation. A suitable deterministic suite should include at least 256 fixed cases covering asymmetric Fourier embeddings, nonuniform phase maps, warps, masks, and boundary offsets.

### Topology and origin

- Degree \(+1\): at least 95% over the noisy/unseen-resampler suite and 100% on noiseless supported fixtures.
- Combined degree \(0,-1,+2\), half/double/higher-harmonic lock: at most 5% suite-wide; no failing individual track may emit a certified target.
- Phase/state collision rate: at most 1% suite-wide; any certified natural track with a detected collision abstains.
- Per-edge phase change: \(-10^{-12}\le\delta_t<0.25\) for an exact certified track.
- At least one uninterrupted complete progress interval.
- Per-cycle circular resultant \(|c_j|\ge0.95\).
- Aggregate resultant \(\rho\ge0.95\).
- Per-cycle landmark-phase coherence
  \[
  \left|K^{-1}\sum_j z(\phi_j^{\max})\right|\ge0.99.
  \]
- With projection normalized by its \(Q_{95}-Q_{05}\) range, the best score must exceed every score outside a \(0.10\)-cycle neighborhood by at least \(0.10\).
- The high-score set within \(0.05\) normalized score units of the maximum must be one component of diameter at most \(0.05\) cycles.
- Circular mean origin must lie within \(0.05\) cycles of the unique peak phase.
- Leave-one-cycle-out origin variation: at most \(0.01\) cycles.

### Gauge, resampling, and repetition

- Rotate teacher phases by a fixed grid of at least 64 angles. Canonical phase error must be at most \(10^{-10}\) cycles and pulse arrays must be bit-identical.
- Test 32, 48, 64, 96, 128, and 256 samples per cycle with smooth and piecewise warps having local speed ratios from \(1/4\) to \(4\).
- At matched physical states, canonical origin error must be at most \(0.005\) cycles and canonical phase error at most \(0.01\) cycles.
- Complete-cycle counts must be exactly identical across resamplers; physical crossing-location error must be at most \(0.01\) cycle.
- Repeat identical cycles \(K\in\{1,2,3,7\}\), with boundary tails of \(0,0.2,0.8\) cycle. After incomplete tails are excluded from origin estimation, exact duplicates must change origin by at most \(10^{-10}\).
- Clips spanning \(0.1,0.25,0.5,\) and \(0.9\) cycle must abstain 100%.

Include a deliberate “clock-blind but warp-variant” attack using velocity magnitude or fixed-frame lags. It should fail; otherwise the test is not exposing the relevant shortcut.

### Symmetry, intersections, masks, and clocks

- Two equal separated projection maxima, antipodal maxima, broad plateaus, and near-ties below the margin: 100% abstention.
- \(q\)-fold symmetric delay-state orbits: 100% topology abstention.
- Instantaneous figure-eight pose with a resolving delay embedding may pass.
- A self-intersection remaining in the complete masked delay state must abstain.
- Missing projection support at the landmark, a temporal gap at the seam, or a gap capable of hiding a traversal: abstain.
- Exact duplicate clocks with identical observations: identical origin and pulse output after collapse.
- Conflicting duplicate clocks: abstain.
- Full time reversal: 100% orientation abstention.
- Pauses at at least 16 evenly spaced phases and lengths \(1,4,16,64,128\): no origin change beyond \(0.005\) cycle and no extra pulses. Any phase excursion or pause-entry/exit pulse duplication fails the teacher.

### Pulse/NOLA/decoder gate

Use events at every window offset, all grid shifts \(0,\ldots,31\), track boundaries, and spacings \(4,5,8,16,32,64,127,128,129\) valid edges. Include plateaus, pauses, padding, duplicate clocks, split attacks, and merge attacks.

For every supported pack:

- Each target event has exactly one supra-\(0.5\) component intersecting its allowed \(\pm1\)-edge neighborhood.
- No component corresponds to more than one target.
- No extra component exists.
- Certified event core satisfies \(R_e\ge0.60\).
- Certified separating valleys satisfy \(R_e\le0.40\).
- Decoder count is exactly invariant under all 32 grid shifts.
- Duplicate-clock and padding edges contribute no support.
- Decoder call count equals the number of complete identities exactly.

The existing allowance of a 1% negative-edge rate is not enough for a per-track exact peak guarantee. Population thresholds may screen a teacher artifact, but each emitted certified track must independently satisfy the stricter crossing tests.

## 4. Counterexamples and kill conditions

1. **Uniform sample-weighted mean.** Oversampling one side of the landmark moves the circular mean without changing the physical orbit. This breaks warp and pause invariance.
2. **Warp-variant delay state.** Fixed-frame lags and velocity magnitude change under monotone resampling despite having no explicit clock input.
3. **Repeated or symmetric maxima.** Two antipodal maxima give \(C=0\); unequal near-ties give a nonzero but perturbation-sensitive origin. Concentration alone is insufficient.
4. **General phase homeomorphism.** Degree \(+1\), orientation, and reconstruction admit \(H\circ f\) for any orientation-preserving circle homeomorphism \(H\). A finite-width circular mean is not invariant to this gauge.
5. **Self-intersection.** If two primitive phases have the same complete masked delay state, no memoryless teacher can distinguish them.
6. **Observable symmetry versus human grouping.** If
   \[
   y(\phi+1/q)=y(\phi),
   \]
   the observable primitive orbit completes \(q\) times during one human-labelled action. No origin rule recovers the human grouping.
7. **Partial clip ambiguity.** The same observed arc can be completed by two unseen orbits with different global projection maxima and different boundary counts.
8. **Hidden-cycle alias.** A pause and one or more complete cycles can have identical sampled endpoints and identical \(S^1\) phases. No endpoint alias loss can distinguish them.
9. **Missing gap.** An arbitrary number of repetitions may occur inside an invalid gap. Splitting components prevents false joins but does not recover the omitted count.
10. **Time reversal.** The origin may remain the same, but phase increments change sign. The stipulated positive half-open crossing rule is not reversal-invariant.
11. **Pause off-orbit transient.** Signed velocities and lags can create a separate transient trajectory on pause entry/exit; the primitive-orbit theorem does not cover it.
12. **Static-code deformation.** If content code changes induce more than a constant rotation, per-track canonicalization does not align teacher targets.
13. **NOLA miss.** Disjoint one-edge predictions from three windows may each average below \(0.5\).
14. **NOLA merge.** Broad responses can bridge the valley between consecutive teacher events, producing one component.
15. **Mask-driven landmark.** A varying mask can become the projection maximum even when the physical pose is unchanged.

Any of the following is terminal for the strict route rather than a cue to tune on human counts:

- nontrivial orbit symmetry or unresolved delay-state collision;
- no complete valid traversal;
- nonunique/unstable projection landmark;
- origin resultant or warp/repetition invariance below threshold;
- unsupported source-clock gaps or inability to defend the no-hidden-cycle sampling assumption;
- backward seam crossings, time reversal, or pause transients;
- any split, merge, or miss in the held-out threshold-\(0.5\) pulse gate;
- teacher-only natural mismatch showing that the observable primitive orbit is not the evaluator’s repetition unit.

## 5. Scientific defensibility of the one-final-peak contract

The contract remains scientifically defensible only in this conditional form:

> For a gap-free, sufficiently sampled, forward traversal of an identifiable primitive masked-delay-state orbit with a unique frozen pose landmark, the system emits one final threshold-\(0.5\) connected component per observed orbit traversal after NOLA.

It is not defensible as an unconditional claim of one peak per human repetition. The remaining human conventions are fundamentally unidentifiable without labels or semantic assumptions:

- whether left/right subcycles are one or two repetitions;
- whether several primitive orbit traversals form one named action;
- where a repetition begins;
- whether boundary partials count;
- whether a pause terminates, suspends, or restarts a repetition;
- how mirrored or symmetric states are grouped.

The canonical projection supplies an algorithmic convention, not evidence that this convention matches annotators. Therefore the route may support a kill-oriented pilot only if it reports abstention/coverage, validates teacher-only counts without tuning, and treats a convention mismatch as terminal. If the benchmark requires a count for every finite, gappy, possibly partial track, the strict theorem does not cover the benchmark and the one-final-peak claim is not presently scientifically complete.
