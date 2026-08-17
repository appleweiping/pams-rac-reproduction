# Experiment Execution Contract: WARP-PHASE Pilot v1

> **NORMATIVE LOCAL-SPECIFICATION CLOSURE**
>
> **PROSPECTIVE PILOT — ZERO ELIGIBLE RESULTS — SAME-FAMILY PROVISIONAL**
>
> **FINAL RESEARCH-REFINE VERDICT: `REVISE` — K8: `BLOCKED / NOT_PASSED_BLOCKED`**

**Date:** 2026-08-15  
**Binding:** `refine-logs/FINAL_PROPOSAL.md` plus this contract  
**Scope:** deterministic implementation and execution semantics only

## 1. Authority and non-expansion rule

This document normatively closes the three bounded local execution blockers in the Round-5 review. It does not replace the Problem Anchor, add a model, add a job, add a baseline, add an endpoint, or add a visible validation block. Where `FINAL_PROPOSAL.md` is complete, it remains authoritative. Where Sections 3.6, 6.2, or 11 left more than one executable interpretation, the smallest deterministic constant below is the binding implementation choice. That choice is an execution binding, not a new scientific contribution.

No gate has run. No pack, prediction, metric, checkpoint, or efficacy result exists. Passage is never inferred from this document. Historical v41/v42/v45/v46/v62/v63, sealed/test material, and existing `results/**` artifacts are ineligible. Fixture outputs are numerical test oracles, not results.

The only prospective study question is whether the signed local warp-integral objective improves supplied-track partial-cache tempo-drift counting relative to identical augmentation without that objective. Prohibited claims include first, SOTA, end-to-end, annotation-free, predicted-track, official MultiRep main result, or deployment readiness.

## 2. Frozen program boundary and storage roots

Implementation must be isolated under `src/pams/warp_phase/` with exactly these planned modules:

- `__init__.py`, `config.py`, `types.py`, `packing.py`, `data.py`
- `selector.py`, `warp.py`, `model.py`, `losses.py`, `decode.py`
- `training.py`, `evaluator.py`, `gates.py`, `cli.py`

The only permitted edit outside that package is attaching a `warp-phase` Typer group in `src/pams/cli.py`. The new path must not call the historical `data`, `train`, or `evaluate` command groups, and historical APIs and configuration fingerprints must remain unchanged. The pilot must reject use of `PoseSequence`, `PAMSEncoder`, `PAMSTCCLoss`, `MultiExpertCounter`, `CountResult`, server v41/v42/v45/v46/v62/v63 code, and Count-main Hann fusion as a base implementation.

The planned configuration is `configs/experiments/warp_phase_pilot_v1.yaml`. The run root has three non-overlapping roots:

```text
data/warp_phase_pilot_v1/
  features/{train,val}/{opaque_key}.{slot}.npz
  vault/
  audit/
```

`features/` contains only `motion`, `person_mask`, `frame_mask`, `sampled_frame_indices`, `source_length`, `opaque_sample_key`, and `local_person_slot`. Persist numeric values as explicit little-endian fixed-width arrays (`motion <f4`, masks `|u1`, clocks/length/slot `<i8`) and the lowercase hexadecimal opaque key as fixed `|S64`; object or Unicode-object arrays are forbidden. ZIP member names are inspected before `numpy.load`; object dtypes, extra members, path traversal, duplicate members, executable formats, and pickles are rejected. Loading uses `allow_pickle=False`. Writers sort member names by Unicode code point and freeze ZIP timestamps, permissions, compression method, and metadata so identical inputs produce byte-identical shards.

`vault/` is evaluator-only and contains joined positive counts and raw integer `[s,e)` periods. `audit/` contains the join, source components, eligibility, hashes, order receipts, and gate receipts. The training CLI accepts only `features/`; its import graph must not reach `evaluator.py`, `vault/`, or an evaluator reader. Opaque keys and local slots may route records and state only and never enter a tensor or learned feature.

Before trusted unpickling, `packing.py` verifies the exact canonical v44 train/validation paths and SHA-256 values from `PILOT_DATA_SCHEMA_AUDIT.md`, rejects every forbidden path, and records the verification. The hash check precedes the first pickle opcode. Gate 0 also requires the 617-byte schema fixture and SHA-256 `31f81549f9c5dbedc6ddb278e868793246826a234de90c06c169c71387428275` exactly.

## 3. Closure A — real-track weak rejection view

### 3.1 Global order and stream

Gate 2 uses one and only one stream:

```python
numpy.random.Generator(numpy.random.PCG64(20270815))
```

Eligible feature shards are ordered by their verified shard SHA-256 interpreted as 64 lowercase hexadecimal ASCII characters. A tie is broken by ascending `local_person_slot`, then ascending opaque key by unsigned UTF-8 byte order. The slot and opaque key occur only in the restricted audit ordering process. Within a track, canonical parent starts are ascending numeric starts from the frozen `0,64,...,K-128` rule. No Python `hash`, per-worker RNG, per-track seed, per-parent seed, framework RNG, or parallel draw stream is permitted. Gate-2 draw generation is single-process and sequential; selector evaluation may parallelize only after all perturbation bytes and their order receipt are frozen.

### 3.2 Per-parent draw and application order

For every parent, let `T` be the exact retained-clock length of that canonical parent (`64 <= T <= 128`). Regardless of missing, invalid, conflicted, or later rejected entries, consume in order:

1. `jitter_raw = rng.normal(0.0, 0.01, size=(T,17,2))`, float64, C-order;
2. `dropout_raw = rng.random(size=(T,17))`, float64, C-order.

Clip jitter elementwise to `[-0.03,0.03]`. Form all canonical joint/frame masks first. Add clipped jitter only to normalized canonical-valid `x,y`; invalid draws are consumed and ignored. Confidence is unchanged. Then drop a canonical-valid joint iff its matching `dropout_raw < 0.10`. For each frame, scan joint indices `0..16` and restore the lowest-index dropped canonical-valid joints until the number retained is `min(8, canonical_valid_joint_count)`. A restored joint retains its jittered coordinates. After the weak mask is final, set all three values of every invalid joint to exact zero. Clocks never change.

The unjittered canonical clean selection is authoritative. The weak view and first/last 64-clock selections are rejection-only. They cannot vote, average, replace, correct, or alter a canonical period.

### 3.3 Receipts

For every parent, persist the following lowercase SHA-256 values:

- raw little-endian float64 C-order `jitter_raw` bytes;
- raw little-endian float64 C-order `dropout_raw` bytes;
- final little-endian float32 C-order weak pose bytes;
- final `uint8` C-order weak joint-mask bytes.

The full order receipt is canonical JSON: UTF-8, no BOM, recursively sorted keys, separators `,` and `:`, and no terminal newline. Each row records ordinal, shard SHA-256, local slot, opaque audit key, parent start, `T`, the four hashes above, and the authoritative/rejection roles. Hash the complete receipt before selector evaluation. Missing rows, reordering, draw-count mismatch, or a hash mismatch invokes K5; there is no redraw.

## 4. Closure B — canonical 1,000-fixture generator and environments

### 4.1 Environment separation

Canonical fixture bytes are generated only in a dedicated environment with CPython `3.12.13` and NumPy `2.4.6`. The project training/evaluator environment remains NumPy `1.26.4`. Training code may load and verify canonical fixture bytes but may never import or invoke the generator or regenerate any expected array.

Before generation, `data/warp_phase_pilot_v1/audit/gate1/fixture_environment.lock.json` must be frozen as canonical JSON and must contain the CPython implementation/version, interpreter artifact SHA-256, OS/architecture/endianness, NumPy version, exact wheel filename and SHA-256, installer version, generator source SHA-256, and its own detached receipt SHA-256. The corresponding training lock at `data/warp_phase_pilot_v1/audit/gate1/training_environment.lock.json` binds `pyproject.toml` SHA-256 `14b3b35bbfa57fc982e24ef2f999df288893c0f0d12a53a9f82120babefead38`, `uv.lock` SHA-256 `5a964f1cd841948eae09ef253098b0b16eb5d2e6bd660e2d3496b389c4dc72a7`, and NumPy `1.26.4`. Empty or unhashed lock fields fail Gate 1.

A fresh `PCG64(20270815)` seeded-kernel witness calls `bit_generator.random_raw(16)` once and serializes the returned values as raw little-endian `uint64` C-order bytes. The required values are:

```text
9521446988715047786,14249219314912930443,6690348893508556226,
11556683178623268117,13450199692548364832,4748815881888473085,
12717261861082187763,6192873454472717068,5919130353202294623,
13700469945621656742,8320809674766470240,14314247923956437868,
13599109478297638902,16225517010854040016,12798711397771314810,
1237992528928537918
```

The 128-byte witness SHA-256 is `22cf6053f9a31cee979636aa1f1f629b890a4a88956d42c99d5bfb4333e3bcc0`. Both environments must reproduce it before any fixture or K4 computation.

### 4.2 Fixed categories

Category membership is assigned before any draw and never changes:

| IDs | Count | Code | Category |
|---|---:|---:|---|
| 0–149 | 150 | 0 | fundamental |
| 150–249 | 100 | 1 | three-segment warp |
| 250–374 | 125 | 2 | symmetric half-harmonic challenge |
| 375–499 | 125 | 3 | symmetric double-harmonic challenge |
| 500–599 | 100 | 4 | off-grid/off-bin |
| 600–699 | 100 | 5 | pause |
| 700–799 | 100 | 6 | reversal |
| 800–899 | 100 | 7 | alias/boundary |
| 900–999 | 100 | 8 | endpoint maxima |

Exactly 250 fixtures, IDs 250–499, are symmetric.

### 4.3 Complete scalar/array draw order

Use one `Generator(PCG64(20270815))` for all IDs in ascending order. Every float draw is float64. Every integer draw uses NumPy `integers(low,high)` with exclusive `high`. For each ID, consume exactly in this order:

1. scalar `phi = rng.uniform(0.0, 2*pi)`;
2. scalar `A = rng.uniform(0.08,0.20)`;
3. `sx = 2*rng.integers(0,2,size=(17,),dtype=int64)-1`, C-order;
4. `sy` by the identical call and shape;
5. category parameters in Section 4.4, including every rejected retry;
6. scalar `h2 = rng.uniform(0.0,0.20)`;
7. scalar `h3 = rng.uniform(0.0,0.10)`;
8. `frame_u = rng.random(size=(128,))`, float64 C-order;
9. `joint_u = rng.random(size=(128,17))`, float64 C-order;
10. `weak_jitter = rng.normal(0.0,0.01,size=(128,17,2))`, float64 C-order;
11. `weak_dropout = rng.random(size=(128,17))`, float64 C-order.

All arrays are fully consumed even for entries later invalidated or ignored. Symmetric templates consume `h2` and `h3` at the same positions but override them as specified below. A retry consumes its complete attempt and never rewinds.

### 4.4 Category parameters and time maps

Let `q = numpy.arange(128,dtype=float64)` and `P_sem` denote the serialized semantic period.

- IDs 0–149: draw `P = integers(12,49)`; set `P_sem=float64(P)` and `c(q)=q`.
- IDs 150–249: draw `P = integers(12,49)`, then draw one accepted schedule below; set `P_sem=float64(P)` and `c(q)=tau(q)`.
- IDs 250–374: draw `n = integers(8,25)`; set even `P=2*n`, `P_sem=float64(P)`, and `c(q)=q`.
- IDs 375–499: draw `P = integers(8,25)`; set `P_sem=float64(P)` and `c(q)=q`.
- IDs 500–599: draw `P = uniform(12.25,48.75)` until `abs(P-numpy.rint(P)) >= 0.20`; at most 128 scalar draws, otherwise fail. Set `P_sem=P` and `c(q)=q`.
- IDs 600–699: draw `P=integers(12,49)`, `a=integers(32,65)`, then `h=integers(8,25)`. Set `c(q)=q` for `q<a`, `c(q)=a` for `a<=q<=a+h`, and `c(q)=q-h` for `q>a+h`. Set `P_sem=float64(P)`.
- IDs 700–799: draw `P=integers(12,49)`; set `P_sem=float64(P)` and `c(q)=127-q`.
- IDs 800–849: draw `P=uniform(4.05,4.45)`; IDs 850–899 draw `P=uniform(62.55,62.95)`; set `P_sem=P`, `c(q)=q`.
- IDs 900–949: no category RNG draw; set `P_sem=4.0`, `c(q)=q`. IDs 950–999 similarly set `P_sem=63.0`.

For a three-segment schedule, each attempt draws in order `b1=uniform(0.20,0.40)`, `b2=uniform(0.60,0.80)`, `log_r=uniform(log(0.5),log(1.5),size=(3,))`, and `pause_u=random()`. Let `r=exp(log_r)` and set `r[1]=0.0` iff `pause_u<0.20`. With segment lengths `d=(b1,b2-b1,1-b2)`, compute `m=r/sum(d*r)` in float64. Accept a non-pause attempt iff all three final slopes lie in closed `[0.4,2.0]`; accept a pause attempt iff `m[1]==0.0` exactly and `m[0]` and `m[2]` lie in closed `[0.4,2.0]`. Reject non-finite or zero normalization. Try at most 128 complete attempts, then fail.

For `x=q/127`, define the continuous endpoint-normalized map `F(0)=0` with slopes `m` on `[0,b1)`, `[b1,b2)`, `[b2,1]`, using the right segment at an internal breakpoint. Set `tau(q)=127*F(q/127)`. This produces `tau(0)=0` and `tau(127)=127` within float64 arithmetic. A scheduled global reversal, when requested outside the category generator, is exactly `127-tau(q)` and is never sign-canonicalized.

### 4.5 Pose equations and symmetric override

For joint `j=0..16`, define `b_x=(j%5-2)/4` and `b_y=floor(j/5)/4` in float64. Confidence begins as `0.9`. For every non-symmetric ID, let `theta=2*pi*c(q)/P_sem` and compute:

```text
x[q,j] = b_x[j] + A*sx[j]*(sin(theta+phi) + h2*sin(2*theta+phi) + h3*sin(3*theta+phi))
y[q,j] = b_y[j] + A*sy[j]*(cos(theta+phi) + h2*cos(2*theta+phi) + h3*cos(3*theta+phi))
```

For symmetric IDs, create `sx_sym,sy_sym` from the consumed signs. For bilateral pairs `(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)`, the lower-index signs are authoritative: `sx_sym[right]=-sx_sym[left]` and `sy_sym[right]=sy_sym[left]`. Indices 0–4 are unchanged. The consumed right signs remain consumed but are ignored.

For IDs 250–374, let `theta=2*pi*q/P_sem` and override the entire generic harmonic template:

```text
x = b_x + A*sx_sym*(0.35*sin(theta+phi) + 1.0*sin(2*theta+phi))
y = b_y + A*sy_sym*(0.35*cos(theta+phi) + 1.0*cos(2*theta+phi))
```

For IDs 375–499, let `theta0=pi*q/P_sem` and use:

```text
x = b_x + A*sx_sym*(1.0*sin(theta0+phi) + 0.35*sin(2*theta0+phi))
y = b_y + A*sy_sym*(1.0*cos(theta0+phi) + 0.35*cos(2*theta0+phi))
```

Thus symmetric IDs consume but do not use `h2,h3`; no unspecified residual harmonic remains. The composition is base skeleton, semantic clock, category time map, exact harmonic template, missingness, then weak rejection view.

### 4.6 Missingness and weak view

If the ID's last decimal digit is in `{0,1,2,3,4}`, set frame-missing threshold `p_f=0.40` and joint-missing threshold `p_j=0.30`; otherwise set both to `0.10`. A frame is missing iff `frame_u[i] < p_f`. A nonsymmetric joint is missing iff `joint_u[i,j] < p_j`.

For symmetric fixtures, each bilateral pair uses only the lower-index uniform: both members are missing iff `joint_u[i,left] < p_j`; the consumed right uniform is ignored. Indices 0–4 use their own uniforms. Frame missingness overrides all joints. Build the canonical mask first, then set invalid canonical `x,y,confidence` to exact float64 zero. There is no equality-inclusive threshold and no post-hoc mask repair.

The weak view clips all `weak_jitter` entries to `[-0.03,0.03]`, adds them to canonical-valid `x,y`, and drops canonical-valid joints where `weak_dropout < 0.10`. It then restores lowest indices exactly as Section 3.2. Confidence stays `0.9` for retained canonical-valid joints. The canonical clean path is never jittered and is always authoritative; the weak path is rejection-only.

### 4.7 Expected serialization and freeze

Write one canonical `metadata.json` and NPY version-2.0 arrays. JSON is UTF-8/no-BOM, sorted keys, separators `,` and `:`, no terminal newline. NPY arrays are C-order with explicit little-endian dtypes. Required filenames and schemas are:

- `q.npy` `<f8[128]`; `pose.npy` `<f8[1000,128,17,3]`; `joint_mask.npy` `|u1[1000,128,17]`;
- `weak_pose.npy` `<f8[1000,128,17,3]`; `weak_joint_mask.npy` `|u1[1000,128,17]`;
- `category.npy` `<u2[1000]`; `semantic_period.npy` `<f8[1000]`;
- `candidate_period.npy` `<u2[125]`, exactly integers 4–128;
- `expected_fft_x.npy` `<f8[1000,256]`; `expected_fft_mask.npy` `|u1[1000,256,34]`;
- `expected_fft_power.npy` `<f8[1000,127]`, bins 1–127;
- `expected_acf.npy`, `expected_score.npy` `<f8[1000,125]`, with exact zero where invalid;
- `expected_score_valid.npy`, `expected_local_max.npy` `|u1[1000,125]`;
- `expected_harmonic_power.npy` `<f8[1000,125,3]`, with exact zero outside a valid candidate;
- `expected_selected_period.npy`, `expected_weak_selected_period.npy`, `expected_first64_selected_period.npy`, `expected_last64_selected_period.npy`, and `expected_reversal_selected_period.npy` as `<i2[1000]`, using `-1` as the only failure sentinel;
- `expected_parent_accept.npy` `|u1[1000]`.

All expected arrays are computed by the frozen selector in the dedicated environment, before Gate 2. Filenames are sorted by Unicode code point. The pack SHA-256 is over `filename_utf8 || 0x00 || file_bytes || 0x0a` for every sorted file. Freeze generator-source and pack hashes together. No fixture, expected value, generator, threshold, or byte may be regenerated or changed after that receipt. Any failure kills K5; it does not authorize a repaired pack.

The training environment must load and hash every canonical file, then independently recompute selector arrays for exactly IDs `[0,150,250,375,500,600,700,800,900,999]` and the K4 vectors. Selected-period integers, valid masks, local-max masks, and category classifications must match exactly. Float64 recomputations use `rtol=1e-12`, `atol=1e-12`, `equal_nan=False`; K4 bytes must match exactly 4,096 bytes and SHA-256 `5e2f35a11bd8a376e44d5ad7d3e5067a0bb5a03e2d61edd84f5f5bfd6cf351ec`. Failure occurs before Gate 1.

Gate-2 thresholds remain unchanged: canonical/rejection agreement within 10% for at least 950 of 1,000 fixtures; half and double classifications each at most 20 of 1,000 overall; and each at most 12 of the 250 symmetric fixtures. Comparisons are inclusive at 10%, 95%, 2%, and 5%. Every eligible real training identity must produce a selector answer; there is no per-track exclusion after eligibility.

## 5. Closure C — exact K4 constructors

### 5.1 Fixed-code fractions

Primary treatment and augmentation-only receive no fixed code. Only the frozen clock/no-pose and mask-only controls receive the already specified four-vector code.

After duplicate-clock collapse and conflict invalidation, before any right-padding, define for retained clock `i`:

```text
f_i = sum_j canonical_joint_valid[i,j] / 17.0
f_track = sum_i canonical_feature_frame_valid[i] / number_of_retained_nonpadding_clocks
```

The denominator includes every retained non-conflict, nonpadding clock, whether its frame is feature-valid or not. It must be greater than zero; otherwise invoke K7. `f_track` is computed once and is constant across every clock of that track. Padding contributes to neither numerator nor denominator. Codes and pose zeroing remain exactly as in the proposal; primary arms never receive `q`, `L`, `f_i`, or `f_track` code.

### 5.2 Matched-time pose shuffle

This is evaluator-only and adds no training job. For each video and exact integer source clock after duplicate collapse, collect eligible supplied identities whose canonical frame at that clock is feature-valid. Order recipients by verified feature-shard SHA-256, breaking ties by local slot and then opaque audit key. If there are at least two, recipient `r` receives the pose values and joint mask of ordered donor `(r+1) mod n`; this is the only rotation direction. If there is exactly one, set that recipient's pose invalid and all pose values to exact zero. If none, do nothing.

Keep the recipient's `q`, `source_length`, identity order, person validity, and routing identity. Recompute frame, cell, support, and padding masks from the donated joint mask under the frozen canonical rules. Never donate a clock, code, state, prediction, label, schedule, or evaluator value. Apply the byte-identical corruption and the same frozen warp schedules to treatment/control and all three seeds.

Bind per-video little-endian float32 pose bytes, uint8 joint/frame/cell masks, donor-index map, and an aggregate canonical JSON receipt to SHA-256 before inference. Coverage is

```text
donor_changed originally valid person-clocks / all originally valid person-clocks.
```

`donor_changed` means the donor's `(shard_sha256,slot,opaque_audit_key)` tuple differs from the recipient tuple. Require coverage `>=0.80`; otherwise K4 fails before computing `P_shuffle`. No random seed or alternate permutation exists.

### 5.3 Unseen midpoint resampler

This evaluator-only constructor has no RNG. For `j=0..319`, compute in float64:

```text
u_j = clip((j + 0.5) * L / 320 - 0.5, 0, L - 1)
```

Query the duplicate-collapsed canonical source track. At an exact stored clock, copy `x,y,confidence` only for a valid non-conflict joint. Strictly between clocks, interpolate each scalar in float64 only when both bounding joints are valid and the bounds are one valid non-conflict source cell, using the frozen warp formula `(1-alpha)*left + alpha*right`. Do not cross conflicts, extrapolate, use nearest neighbor, or repair a joint. Form masks first, cast valid scalars to float32 ties-to-even, then zero invalid entries. Derive frame and cell masks under the same hip/eight-joint/support rules. The float64 `u` grid is audit/evaluation metadata and is never a model feature or fixed code.

Hash the grid, resampled pose/masks, and the already frozen warp-schedule bytes. Treatment/control and all seeds use identical resampler and schedule bytes. The midpoint tensor is constructed once per eligible track and is never redrawn or selected by a metric.

### 5.4 Unchanged K4 decision interface

Let `G=M_c-M_t`; Stage B is reached only when `G>0`. All metrics are finite, unclipped, seed-averaged normalized video-first AvgMAE on identical eligible units.

| Check | Statistic | Fail iff |
|---|---|---|
| clock/no-pose | `Q_clock=(M_c-M_clock)/G` | `Q_clock>0.20` |
| mask-only | `Q_mask=(M_c-M_mask)/G` | `Q_mask>0.20` |
| matched-time shuffle | `P_shuffle=(M_c^sh-M_t^sh)/G` | `P_shuffle>0.20` |
| unseen resampler | `P_unseen=(M_c^u-M_t^u)/G` | `P_unseen<0.80` |

No clipping, absolute value, epsilon, alternate denominator, confidence interval, preservation synonym, or post-hoc threshold is allowed.

## 6. Gates and architecture acceptance

### Gate 0 — pack/vault/isolation

Gate 0 is deterministic CPU work and creates no training job. It must verify forbidden paths before pickle access, hash before trusted unpickle, exact whitelist and deterministic NPZ bytes, feature/vault permission separation, count-blind eligibility, all nine original development components nonempty, and import-graph separation. Any failure invokes K6 or K7.

### Gate 1 — numerical/environment fixtures

Gate 1 is deterministic CPU work and creates no training job. It must pass the dual-environment lock and witness, every fixture byte/hash, the fixed 10-case cross-environment selector recomputation, duplicate collapse, root/scale, FFT/ACF, warp tensor, pause/reversal/alias, signed overlap, recurrence, optimizer accumulation, NOLA/decode, K4 bytes/routing, period extraction, normalized AvgMAE, and bootstrap fixtures. Fixtures are receipts, not efficacy.

### Gate 2 — selector stability

Gate 2 is deterministic CPU work and creates no training job. It runs the immutable 1,000 pack and every eligible training identity through the single selector using the frozen real-track stream/order. Any global threshold, real-track answer, perturbation hash, or order-receipt failure invokes K5.

### Gate 3 — CPU/tiny-GPU sanity

Only after Gates 0–2 pass may a CPU smoke check and at most three tiny-GPU training-only sanity jobs run. They test finite gradients, loss decrease, state hold/single commit, deterministic receipts, identity isolation, and prediction schema without opening count or period labels. These three jobs are part of the 59-job ceiling and together cap at 6 GPU-hours.

Stages A–E retain the proposal's exact conditional order. Full jobs cap at 12 GPU-hours, tuning jobs at 3, and total training at 59 jobs/552 GPU-hours. No stage may start when a dependency or K-rule fails.

## 7. Frozen K1–K9 and reporting surface

- **K1:** pass only if `R>=0.05` and the paired all-nine-component absolute-effect 95% CI lower endpoint is strictly positive; `M_c=0` fails.
- **K2:** a reached closest prior matching/exceeding treatment under the frozen paired rule kills the residual claim.
- **K3:** deletion, direct signed-frequency, global-only, or unsigned-reversal parity kills objective necessity.
- **K4:** use only Section 5.4 plus receipt/finite/parameter/budget integrity.
- **K5:** any fixture, selector, harmonic, coverage, recurrence, NOLA, or execution failure kills the phase-mass claim; no harmonic correction.
- **K6:** any isolation, source, permission, forbidden-field, prediction-freeze, or evaluator-integrity failure stops with no claim.
- **K7:** any empty original development component, missing scale, invalid population/clock/support, or zero retained denominator stops or restricts work to engineering diagnostics.
- **K8:** remains `BLOCKED / NOT_PASSED_BLOCKED`; claim freeze remains blocked until complete TWCRAC primary-source adjudication.
- **K9:** masked geometric TSSM can never become an automatic fallback contribution.

There are exactly three visible validation blocks: primary paired law versus augmentation; staged mechanism/shortcut/closest-prior falsification; and identity isolation. Gate receipts are supplementary engineering evidence and never a fourth block.

## 8. Authorization state

This contract closes the three Round-5 blockers at the specification level only. Actual lock files, fixture bytes, feature/vault/audit packs, and receipts must still pass Gates 0–2. Training is therefore **BLOCKED** now. The final verdict remains **`REVISE`**, the evidence count remains zero, the work remains a partial supplied-track pilot, and K8 independently blocks novelty and claim freeze.
