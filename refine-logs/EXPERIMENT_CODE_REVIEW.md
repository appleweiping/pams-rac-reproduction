# WARP-PHASE Pilot v1 Experiment Code Review

**Verdict: BLOCKED**  
**Implementation disposition: REVISE**  
**Training/server launch authorized: NO**  
**Reviewer:** `gpt-5.6-sol`, OpenAI family, same-family review; acceptance is **provisional**.

This review is bound to the exact snapshot hashes recorded below and in `EXPERIMENT_CODE_REVIEW.json`. It is a local, read-only scientific and engineering review. No server was contacted; no real train, validation, test, sealed, held-out, or results artifact was opened; and no experiment, pack, training, prediction, or evaluation job was run.

The implementation contains useful fail-closed scaffolding and several correct primitives, but the current immutable fixture candidate is not freeze-valid, it genuinely fails K5/Gate 2, Gate 0–3 cannot authorize training, and the package contains no actual training transition. The passing unit suite does not change that conclusion.

## Blocking findings

### CRITICAL C1 — The exact fixture candidate loses the diagnostic oracle and cannot pass its mandatory cross-environment check

The generator initializes every diagnostic array to zero, calls `select_support_period`, catches `SelectorError` as `None`, and fills diagnostics only when selection succeeds (`scripts/experiments/generate_warp_phase_fixture_pack.py:416-424,435-461,464-472,523-529`). In the exact candidate, all 240 rows whose canonical selected-period sentinel is `-1` have all-zero `expected_fft_x`, `expected_fft_mask`, `expected_fft_power`, and `expected_score_valid`. Those are not valid serialized selector diagnostics; they are the untouched allocation defaults after a late selector rejection.

The contract requires independent recomputation of IDs `[0,150,250,375,500,600,700,800,900,999]`, including selected-period failure sentinels and diagnostic masks (`EXPERIMENT_EXECUTION_CONTRACT.md:190-208`). The verifier calls `select_support_period` without a failure branch (`src/pams/warp_phase/gates.py:743-783`). Exact recomputation yields selector failure for IDs 0, 150, 250, and 900, so Gate 1 raises before it can compare their required `-1` outcomes.

The proposal also requires three deterministic FFT witnesses (`FINAL_PROPOSAL.md:142`). The generator reserves no dedicated witness rows and applies stochastic missingness to every fixture (`generate_warp_phase_fixture_pack.py:483-505`). Direct inspection of the exact pack found no all-valid row at all, hence no all-valid `P=20` witness; six `P=63` rows have support span 126, but their selected periods are `61, 9, 22, 58, -1, 54`, so the required span-126 upper-endpoint witness selecting 63 is absent. The existing `PASS_FREEZE_CANDIDATE` audit is therefore overturned: byte integrity is not semantic validity.

Required resolution: define a result type that preserves diagnostics on a legitimate no-selection outcome, regenerate only under a newly authorized protocol if immutability rules permit it, add explicit deterministic FFT witness rows, and make the ten-case verifier compare both success and failure outcomes. The current bytes must not be frozen as valid.

### CRITICAL C2 — The immutable candidate genuinely fails every Gate 2 aggregate threshold

The exact pack aggregate SHA-256 is `b2c791f9aac1d86c0849fa74b00b4e605fe21120f9d188f4ec0b304d7052f173`; its receipt SHA-256 is `f5e03b431f47f2ae0c1e52f40955fbcb70b97642e52969178ffcfa3140f6c5ce`. The audited results are:

| Gate 2 statistic | Observed | Required |
|---|---:|---:|
| canonical/weak agreement | 246 / 1000 | at least 950 / 1000 |
| overall half classification | 55 | at most 20 |
| overall double classification | 144 | at most 20 |
| symmetric half classification | 50 / 250 | at most 12 / 250 |
| symmetric double classification | 66 / 250 | at most 12 / 250 |

This is a real K5 failure, not merely missing paperwork. The candidate receipt is explicitly non-authoritative, performs no freeze action, and authorizes nothing. No repair, threshold selection, redraw, or silent regeneration is allowed to turn these exact bytes into a pass.

### CRITICAL C3 — Server execution and training launch are not possible from this snapshot

The implementation deliberately reports:

- Gate 0 blocked because the typed v44 adapter, real pack/vault/audit products, component/join receipts, and import-isolation receipt do not exist.
- Gate 1 blocked because the complete numerical witness schema is missing (`gates.py:857-866`) in addition to C1.
- Gate 2 blocked because perturbations are drawn inside selector evaluation and no pre-drawn input API/order receipt exists; the code correctly refuses to open real feature shards (`gates.py:900-958`).
- Gate 3 blocked because the prediction schema and three tiny-GPU receipts are absent (`gates.py:978-1085`).
- `train` is only a feature-inventory preflight that always returns `BLOCKED` with an empty authorization list (`training.py:75-140`). There is no actual train/predict/evaluate command path for this experiment.
- The frozen config says `training_authorized: false` (`configs/experiments/warp_phase_pilot_v1.yaml:5`).

The local preflight report also records zero launch authorization and a candidate server CPython 3.10.15 environment while the contract requires CPython 3.12.13; it additionally requires a fresh output-space receipt before any launch. These are local receipt observations only—this review did not contact the server.

Required resolution: close Gates 0–3 in order, implement the frozen pre-draw interface and complete witness set, freeze prediction/training schemas and receipt chains, bind the approved config, and add an actual authorization-consuming training transition. Do not launch against the current candidate.

## Major findings

### MAJOR M1 — Invalid-frame pose values leak into valid-frame predictions

`WarpPhaseModel._input_channels` masks joints only with `joint_mask & clock_mask`, not with `frame_mask` (`model.py:127-135`). The two kernel-5 temporal convolutions run before the recurrent state-hold decision (`model.py:152-169`). A frame can be feature-invalid under the hip/eight-joint rule while still retaining valid joint values, so those values influence neighboring valid clocks through convolution.

A deterministic probe changed only pose values at one `frame_mask=False` clock while leaving its joint mask true. The invalid output remained zero and the GRU state held, but valid-frame phases changed with maximum absolute difference `1.9523284435272217`. This violates the intended invalid-cell isolation: state hold alone is insufficient.

Required resolution: mask pose and mask channels by `frame_mask & clock_mask` before temporal convolution (or freeze an equally strict alternative), and add a regression asserting that arbitrary invalid-frame pose changes cannot alter any valid-frame output.

### MAJOR M2 — Gate 1 checks environment-lock syntax, not artifact provenance

`verify_environment_lock` validates fields, version strings, and SHA-256 syntax only (`gates.py:469-504`). `gate1_fixture_receipt` hashes the lock itself but does not open/hash its detached environment receipt, rehash the interpreter or NumPy wheel, compare the lock's `source_sha256` to the current generator, or bind the candidate selector/generator hashes to current source (`gates.py:821-850`). The existing audit explicitly says the interpreter and wheel were not rehashed.

Required resolution: verify the detached receipt bytes first, rehash available interpreter/wheel artifacts in the correct environment, and compare every lock/metadata/pack source binding to the exact current source before setting the environment or fixture checks true.

### MAJOR M3 — Blocked gate and training commands return shell success

The gate and `train` CLI handlers serialize their receipts and return without raising a nonzero `typer.Exit` (`src/pams/warp_phase/cli.py:87-209`). Therefore a scheduler can see exit code 0 even when the JSON status is `BLOCKED`. Content-level authorization is fail-closed, but process-level orchestration is ambiguous.

Required resolution: define and test nonzero exit semantics for `BLOCKED`/`FAIL`, while preserving the exclusive receipt as the authoritative structured record.

### MAJOR M4 — The executable interface is incomplete and not bound to the experiment config

The CLI has static preflight, lock, reserve, Gate 0–3 CPU, and blocked `train` commands, but no config argument/fingerprint binding, feature/vault pack command, pre-drawn real-track selector command, actual train/predict/evaluate command, Stage B K4 execution, fixed-code control application, matched-time shuffle, unseen midpoint resampling, scheduler manifest, or end-to-end receipt chain. The configuration is validated in `config.py` but is not consumed by these commands.

Required resolution: implement the contract's complete, authorization-gated command graph and bind every receipt to the exact config file hash/fingerprint before server execution is considered possible.

### MAJOR M5 — Physical feature/vault separation is not enforced as a privilege boundary

The layout uses distinct directories and POSIX modes, which is useful, but explicitly skips Windows ACL enforcement (`packing.py:148-186`). The same process/user can still read both roots. Import isolation reduces accidental coupling but is not the requested evaluator-vault permission separation. In addition, `pack_identity` publishes eligibility and feature artifacts before invoking/validating the vault factory (`packing.py:804-840`), so a vault failure leaves an irreversible partial identity under exclusive-write semantics.

Required resolution: produce an OS-principal/ACL or equivalent permission receipt for the evaluator boundary and stage/validate the whole identity before atomically publishing feature, vault, and audit products.

### MAJOR M6 — Tests validate primitives but miss the decision-bearing failures

The targeted suite passes, but no test runs the exact candidate through Gate 1/2, checks diagnostic arrays on `-1` rows, proves the three mandatory FFT witnesses, exercises failure-sentinel ten-case recomputation, tests invalid-frame convolution isolation, verifies detached environment provenance, or asserts blocked CLI exit semantics. These omissions explain why 118 tests pass while the experiment remains non-executable.

## Minor findings

### MINOR N1 — Gate 2 receipt diagnostics obscure immutable-pack success on threshold failure

`gate2_selector_receipt` sets `immutable_1000_fixture_pack` only after threshold verification returns and catches the entire block by exception type (`gates.py:922-935`). A genuine threshold failure therefore reports the immutable pack check as false and loses the exact scientific failure message. Split integrity/hash validation from scientific threshold evaluation and preserve both outcomes.

### MINOR N2 — Official evaluator parity remains source-receipt provisional

The evaluator correctly and explicitly limits its claim to the temporal supplied-track kernel and disclaims detection/tracking/end-to-end parity (`evaluator.py:338-362`). That boundary is correct. However, the official source hashes are inert constants, the source is not verified at runtime, and the tests state that the pinned upstream repository has no official Period-AP fixture (`tests/test_warp_phase_evaluator.py:177-178`). Keep all parity claims at the narrow, provisional boundary until a primary-source verification receipt exists.

## Confirmed correct in this snapshot

- Source pickle handling is hash-before-first-opcode: the exact allowed path is read into stable bytes, checked against the canonical SHA-256, and only those same bytes are passed to `pickle.loads` (`packing.py:267-328`). No real pickle was opened during this review.
- Feature archives are pre-inspected for exact ZIP members and safe NPY schema and are then loaded with `allow_pickle=False`; deterministic ZIP/NPY writing and whitelist behavior pass the tests.
- The training-side package initializer is dependency-free, and training dynamically imports only the feature reader; evaluator-vault data is not imported into the implemented model/training path.
- The model has exactly **225,026** trainable parameters.
- Optimizer accumulation uses exactly four fresh forward/backward graphs, scales each loss by four, clips once, and performs one AdamW step with gradients cleared on both success and failure.
- Signed overlap, warp interpolation/masking, K1/K4 formula primitives, NOLA reconstruction, half-even decode, count evaluation, and the narrow supplied-track temporal evaluator agree with their targeted deterministic tests.
- Gate and training receipt contents authorize nothing; no tested code path can silently start training.

## Verification performed

- `pytest` over every `tests/test_warp_phase_*.py` plus `tests/test_cli.py`: **118 passed, 1 skipped** in 54.31 s. The skip is the Windows symlink-privilege case.
- Ruff over the experiment package, generator, attachment, and tests: **passed**.
- mypy over `src/pams/warp_phase` and the generator: **passed** for 15 source files.
- Exact fixture diagnostics used `numpy.load(..., allow_pickle=False)` only on the candidate pack; no real data was accessed.
- Mandatory ten-case outcomes: `0=-1, 150=-1, 250=-1, 375=48, 500=22, 600=18, 700=11, 800=8, 900=-1, 999=54`.

## Snapshot bindings

Primary review inputs:

| Path | SHA-256 |
|---|---|
| `refine-logs/FINAL_PROPOSAL.md` | `e8ff129b3a67ac07be0e5ff1af09555bb26f000e7cde9315b13493733cda6418` |
| `refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md` | `70f80912e4b9693fae9e859eb12f90cc754b7d523b59af766879518770d69b2e` |
| `refine-logs/EXPERIMENT_PLAN.md` | `020626027a73939f3fd4dfa479e6592699d19f652da592682b90d9235724ce5c` |
| `configs/experiments/warp_phase_pilot_v1.yaml` | `e59fb34599eaece41545c237b18871b5a206046e53fb1246795f9dcb3d681063` |
| `scripts/experiments/generate_warp_phase_fixture_pack.py` | `8a12a188c95d38f8958553e10ee06df6920c372e78eef2c2de575e7b8d6eb572` |
| `src/pams/warp_phase/selector.py` | `8a318f82ff9e739e3889940a575d8d3594a04349f2cd9881b9f1974ead25ceb2` |
| `src/pams/warp_phase/gates.py` | `be850809457dba85862b45631b89e068d23517699587ea13a0f2179fe44ac6c5` |
| candidate receipt | `f5e03b431f47f2ae0c1e52f40955fbcb70b97642e52969178ffcfa3140f6c5ce` |
| exact candidate pack aggregate | `b2c791f9aac1d86c0849fa74b00b4e605fe21120f9d188f4ec0b304d7052f173` |
| fixture candidate audit JSON | `5e7df8b6f4033b4162fc359cf67628fa00921276428fc1aaad49d92ff3d8e305` |

The complete per-source, per-test, per-pack-member, audit, environment, and local-preflight SHA-256 ledger is in `refine-logs/EXPERIMENT_CODE_REVIEW.json` and the ARIS request trace. Any post-review source change is outside this verdict.
