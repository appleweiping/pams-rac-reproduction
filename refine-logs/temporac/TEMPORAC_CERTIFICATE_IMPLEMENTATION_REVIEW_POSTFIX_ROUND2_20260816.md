# TempoRAC Certificate Implementation Post-fix Review — Round 2

**Date:** 2026-08-16  
**Reviewer:** `gpt-5.6-sol` (`/root/temporac_certificate_postfix_review2`)  
**Review relation:** same-family, provisional  
**Fresh context:** `true`  
**Verdict:** **REVISE**  
**Authority:** P0/P1/P2/P2-METRIC/P3/S0/launch/server/data/training/evaluator/paper-claim = **0**

## Outcome

The post-fix implementation closes the previous deep-immutability defect for certificate, feature, natural-teacher, X0-view, and loaded-target arrays; it revalidates a sealed copy immediately before target serialization; and it closes the previous X0 duck-type/noncanonical-view acceptance. The K7 bridge now consumes typed pulse and source-clock evidence before the one-use capability is marked consumed and its primitive check rejects split/merge count cancellation.

The implementation is nevertheless not gate-clear. Four independently reproduced blockers remain:

1. natural teacher output and the selected-teacher digest are caller assertions rather than outputs derived from an authority-bearing selected teacher/checkpoint;
2. a natural target artifact is not bound to the natural input receipt or feature content that produced it;
3. K7 accepts a complete arm/seed/condition cross-product over an arbitrary identity subset, while reporting a 402-row join and allowing a `PASS` decision;
4. the canonical P05M receipt builder returns a `MappingProxyType` that the evaluator's canonical JSON hasher cannot serialize directly.

No gate, launch, evaluator, natural-data, training, result, or paper-claim authority follows from this review.

## Prior-finding closure matrix

| Prior finding | Round-2 status | Independent evidence |
|---|---|---|
| F1 — deep immutability / pre-serialization mutation | **Closed for the certificate boundary** | `FeatureRecord`, `CertifiedTarget`, `CertificateTrack`, `NaturalTeacherOutput`, X0 views and loaded target arrays reject `flags.writeable=True`; `dataclasses.replace` cannot mint an authorized target; `CertifiedTarget.artifact_arrays()` calls `sealed_copy()` before serialization (`src/pams/temporac/types.py:72`, `src/pams/temporac/types.py:359`, `src/pams/temporac/types.py:398`). |
| F2 — X0 canonical source binding | **Closed** | Exact `X0View` type is required, the inventory is checked, `generate_view` is rerun and all scalar/array content is compared, and the source key is rederived (`src/pams/temporac/certify.py:707`–`755`). Both a duck type and altered canonical content were rejected. |
| F3 — natural feature/teacher binding | **Open, split into R2-F1 and R2-F2** | Feature/preprocess hashes are rederived, but the teacher digest and output arrays remain caller-controlled; the resulting target has no input-receipt commitment. |
| F4 — K7 certificate primitive bridge | **Mechanism present, finding not closed** | Pulse/source-clock and target-count content are consumed before `self._consumed = True` (`src/pams/temporac/evaluator.py:455`–`462`), and exact one-event-per-interval is checked (`297`–`320`). Population completeness remains bypassable (R2-F3). |

## Blocking findings

### R2-F1 — natural teacher authority is not derived (`blocking`)

`build_natural_teacher_output` accepts `selected_teacher_sha256`, `phase`, and `reconstruction` directly (`src/pams/temporac/certify.py:868`–`876`). It validates only digest syntax (`886`–`892`), feature-derived preprocessing dimensions/hashes, and hashes of those caller-provided arrays (`893`–`925`). It never consumes an authority-bearing selected checkpoint, derives its digest from artifact bytes, or runs/verifies `TempoRACTeacher` inference.

The independent probe passed the same feature, phase, and reconstruction under digests `11…11` and `22…22`. Both outputs and both certified targets were accepted; phase, reconstruction, pulse, and chi were byte-identical while teacher provenance changed. This permits arbitrary teacher relabeling without changing scientific content.

**Required closure:** make natural teacher output obtainable only from a typed/authority-bearing selected-teacher artifact or receipt; derive its SHA-256 from exact checkpoint bytes; bind canonical preprocessing bytes; and derive or independently verify the inference arrays under that selected teacher. Add negative tests for arbitrary digest relabeling and raw-array injection.

### R2-F2 — natural target is not bound to its input receipt (`blocking`)

`CertifiedDevelopmentTarget` verifies that feature receipt and natural input receipt agree, then checks only target source kind, opaque key, slot, and teacher digest (`src/pams/temporac/evaluator.py:134`–`164`). The closed target artifact/receipt carries no natural-input-receipt or feature-artifact commitment. No later evaluator step reconstructs the certificate from the supplied natural input.

The independent probe constructed feature A and feature B with the same opaque key, slot, and teacher digest but different motion. Their certified pulse ledgers differed (8 versus 4 events). Pairing feature A and its input receipt with feature B's target artifact was accepted and exposed B's four-pulse ledger as if it belonged to A. This is a content-level target/receipt swap, not a malformed NPZ attack.

**Required closure:** introduce an authority-bearing binding that commits the exact natural input receipt (and selected-teacher output receipt) to the certified target, or rederive the target from those exact inputs at K7. Closed target receipt schemas must be changed only through the contract amendment process. Add a same-identity/different-feature swap test.

### R2-F3 — K7 does not require the committed 402-identity / 9,648-prediction population (`blocking`)

`_validate_prediction_population_before_consume` requires the arm/seed/condition cross-product only for identities present in `predictions`; it has no expected identity set or expected count (`src/pams/temporac/evaluator.py:228`–`251`). `validate_certified_target_commitment` accepts any nonempty evidence subset whose identities occur in that prediction subset and whose locally supplied target-count rows match (`254`–`294`). Neither set is equated to `population_identities`, even though the vault join independently requires 402 identities (`185`–`208`). G5b then hard-codes `join_cardinality: 402` (`472`–`480`) while the metric rows are built only from the supplied prediction subset (`481`–`484`).

The end-to-end independent probe supplied a valid 402-row population/vault but only 8 prediction identities (192 prediction artifacts) and one certified target. `OneUseEvaluator.run` accepted the bundle, emitted G5b join cardinality 402, and returned K7 `PASS`, while G5a's declared artifact count remained 9,648.

**Required closure:** before capability consumption, require prediction identities, certified-development-target identities, target-count identities, population identities, and the committed K1/G5a eligible identity set to be exactly equal; require exactly 402 unique identities and exactly 9,648 unique prediction observations; validate the declared artifact count against observed envelopes; derive G5b join cardinality from the validated equality rather than a constant.

### R2-F4 — canonical P05M builder output cannot be consumed directly (`blocking`)

`count_metric_fixture_receipt` returns `MappingProxyType(payload)` (`src/pams/temporac/evaluator.py:499`–`519`). `OneUseEvaluator.run` accepts a `Mapping[str, object]`, validates it, then sends it directly to `canonical_json_bytes` (`446`–`449`). Python's JSON encoder rejects `mappingproxy`, so the output of the public canonical builder raises `HashIOError` when passed directly to the public evaluator. The independent K7 probe could proceed only by adding an uncontracted `dict(metric_fixture)` conversion.

**Required closure:** make the builder and consumer directly composable, preferably by canonicalizing a validated plain dictionary internally, and add an end-to-end test that passes the builder's return value without caller normalization.

## Attack results

| Attack | Result |
|---|---|
| `dataclasses.replace` on a certified target | rejected — authority cannot be reconstructed |
| `flags.writeable=True` on feature/certificate/natural-output/X0/loaded-target arrays | rejected — immutable bytes backing |
| duck-typed X0 view | rejected |
| exact X0 type with altered canonical array content | rejected |
| arbitrary selected-teacher digest with unchanged output arrays | **accepted — blocker R2-F1** |
| post-construction pulse/chi/mask mutation at certificate boundary | rejected by bytes backing; pre-serialization sealing also revalidates semantics |
| malformed target NPZ or mismatched target receipt | rejected by joint loader and semantic validators |
| same-identity natural target from different feature/input | **accepted — blocker R2-F2** |
| incomplete vault/population join | rejected |
| complete vault join plus prediction/certificate subset | **accepted with K7 PASS — blocker R2-F3** |
| split+merge count cancellation (two pulses assigned to one interval, none to another) | rejected by exact interval ownership check |
| canonical P05M builder output passed directly to evaluator | **rejected by serialization type mismatch — blocker R2-F4** |

One residual hardening issue is not counted as a separate certificate blocker: `TrustedSourceObject` copies arrays but only clears their NumPy writeable flag (`src/pams/temporac/trusted_packer.py:123`–`131`), so callers can re-enable it. The probe confirmed this. Because this is explicitly a privileged trusted-packer input rather than an exposed certificate artifact, it is recorded as a warning; bytes backing or immediate pack-time resealing would remove the footgun before any authority-bearing use.

## Verification

All commands ran in the repository `.venv` (CPython 3.12.13, NumPy 1.26.4, SciPy 1.17.1, Torch 2.13.0+cpu, pytest 8.4.2, Ruff 0.16.3, Mypy 1.20.2). No server, natural-data, dataset-test, sealed, heldout, or results path was opened.

| Check | Result |
|---|---|
| Targeted certificate/evaluator/firewall/preprocess/prediction tests | `62 passed, 1 skipped in 79.56s` |
| Full `tests/temporac` | `115 passed, 1 skipped in 180.77s` |
| `ruff check src/pams/temporac tests/temporac` | PASS — all checks passed |
| `mypy src/pams/temporac` | PASS — 24 source files |
| `mypy src/pams/temporac tests/temporac` | FAIL — one test annotation error at `tests/temporac/test_x0.py:146` (`numpy.float64` passed where `float` is annotated), 43 files checked |
| Independent adversarial probe | PASS as a probe; it reproduced R2-F1/R2-F2/R2-F3/R2-F4 and the trusted-source warning, and confirmed the named rejection properties |

The full Mypy failure is not the scientific provenance blocker, but a clean full-scope type-check claim would be false until the test annotation is corrected.

## Terminal integrity

- Git HEAD at review: `238879a475e3bb1fde24da76a8b3258b7fe68dd3`
- Audited-input manifest SHA-256: `8c0f780347032a81bf90c0e102fba40f2a6570a50562077fa11f6f5f045e9a81`
- Canonical proposal SHA-256: `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391`
- Experiment plan SHA-256: `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e`
- Experiment tracker SHA-256: `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b`
- Research contract SHA-256: `3e037a2f74c49dd62e596dc7b875d94216e45b7f693b80a7afb6a60f8244367b`
- `types.py`: `4249eac2738f5b2edf53f38cbb681dffae609c151d493dd03b625fe10ad4c3ca`
- `certify.py`: `468ce8418bff0f43582d341f3e1b8064cf1c429301d1f56be84b8e13ad59a3f1`
- `evaluator.py`: `f981c6c7061c3388f06ef39466ef98192e65c62c501821be2b211f903cfaf9de`
- Independent probe: `a471d96e122fc4ab51d1bb7da51dec1c4cba324d917ce273f14f91e649dcae2b`

The structured companion contains terminal hashes for every audited source, dependency, and TempoRAC test.

## Authority ceiling

`REVISE` is an implementation-integrity verdict, not a gate result. `authoritative=false`, `authorizes=[]`, and every authority bit remains zero. P0 cannot be promoted and no P1/P2/P2-METRIC/P3/S0, natural-data, training, evaluator-capability, launch, result, or paper-claim transition is authorized.
