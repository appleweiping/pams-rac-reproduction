# TempoRAC Amendment 003 P05M Fixture Review — Round 2

## Verdict

`REVISE`

Status: `REVISE_SAME_FAMILY_PROVISIONAL_ZERO_AUTHORITY`.

Amendment Markdown `d615def9dc468b7e483488b87cb0524302f1c4c325df8bf092e741e21f6262d1` and JSON `c73b85731ece838684f78c41b3dfcdb498af5cf9822f7ee84d2add4d09ffe1fc` resolve the Round 1 historical-hash self-lock and state the correct K7 verify-before-consume order. The fixture arithmetic also reproduces exactly. Four specification-level blockers nevertheless leave P2 evidence non-closed and permit the six-key inner summary to remain authoritative through the unrevised plan/tracker path.

This same-reviewer continuation is same-family provisional. It grants no implementation, S0, gate, capability, launch, data, server, test, training, result, Git, paper, or claim authority.

## Binding and persistence-time supersession

The complete Round 2 read occurred while the canonical proposal/plan/tracker paths still contained the following reviewed bytes:

| Reviewed input | SHA-256 |
|---|---|
| `refine-logs/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` |
| `refine-logs/temporac/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` |
| `refine-logs/EXPERIMENT_PLAN.md` | `3fd1986871b32674d74b395fbfc9720712d9fba0c8583c24c1d5bddb87086eb6` |
| `refine-logs/EXPERIMENT_TRACKER.md` | `c7e2f32c3c81b66884abb13760a671ecc7cdd314fd6c994a4ed4c46cb754a63b` |

Those exact bytes remain independently hashable at `FINAL_PROPOSAL_20260816_041423.md`, `temporac/FINAL_PROPOSAL_20260816_034707.md`, `EXPERIMENT_PLAN_20260816_044644.md`, and `EXPERIMENT_TRACKER_20260816_044644.md`. After the complete read, a separate authorized canonicalization first superseded the canonical paths with proposal `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391`, plan `5876b40075b488866960ab2c74f8aac5f655ef8df8727cdebdf89cc1d4814263`, and tracker `176b0ef18fe0b19ebdbe140638accb2d71fe3742be14a233042de54cd3a5ee71`. During final persistence validation, plan and tracker had advanced again to `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` and `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b`; `contract.py` had advanced from the reviewed `b71255a35cce1701d2741064064b53ba819b742873a348f1852f147149e36467` to `5ab8fb62dc558ba78e95fc07e52c1f1a1afbff5b1c5003834079eea6143d745f`. All superseding bytes are excluded from the Round 2 input set and are not adjudicated here. Amendment 003 itself remained stable.

## Final stable code snapshot

| Input | Bytes | SHA-256 |
|---|---:|---|
| `src/pams/temporac/metrics.py` | 17608 | `1dadf08e89330bed2ab773afda05bffcb3fe2252b0d3e23c0bcc18c2c3b049dd` |
| `src/pams/temporac/evaluator.py` | 13527 | `81037186896309b093cd184b6e63542653f5a7e1ab063b78a9e63558369a47d2` |
| `src/pams/temporac/contract.py` | 9079 | `b71255a35cce1701d2741064064b53ba819b742873a348f1852f147149e36467` |
| `src/pams/temporac/types.py` | 24668 | `32cd9450e836d97d4311d59ad504c987ff47970a134ac1f860d96951caea8228` |
| `src/pams/temporac/hashio.py` | 27213 | `df872dc5709cc8814b5d11f87d4de8cc0351459cf5cdf04e19afd0a2baf2b13c` |
| `src/pams/temporac/receipts.py` | 24707 | `ed1a993a832b5cb471a56ec6c9df39f5f6873efb88a33c223f4a4d741e946f15` |
| `src/pams/temporac/trusted_packer.py` | 17137 | `40d6e3d5be1b26b7ba0f0f7aadf8da7496c95c4e829face5c7e9faaee2cdf94e` |
| `src/pams/temporac/gates.py` | 5929 | `b6c9d7432443d7ad36738a81700746709fd571d475514aa6f1bbcd39be744b4a` |
| `src/pams/temporac/fixtures.py` | 17856 | `a8d10df7c8e3cbcb5684198192d3a553921da0f6aa92f03e54448ee182671406` |

Using the amendment's exact seven-member order and preimage produces 134301 preimage bytes and current root `755743d16190b9d055b0f6a3c341ada4b8a6084e0c49477287bddc2755ad50bd`. This is a reproducible hash of the seven named files; it is not a complete Python executable closure and is not an accepted candidate index.

The review observed and excluded these nonbinding integration snapshots:

1. Round 1 pre-format: `hashio=f6f8e4e6f891c510c561b507d6826bae3cddb12ee9c34d4bade584f98e7fb0bf`, `receipts=5f30581bdf31be4f98a891f9631f0ef7eda9ddcd58c2893f064ab277c3d879bd`.
2. Round 1 stable review: `hashio=2d81e415b45e3d1b612f998b6a8386a967dfdfcfaac31bc914b2c95cbaeeab3a`, `receipts=0ac800d96f536a41a67775bc8e8e778fe14c312c02f642f7cde78d210244016d`.
3. Round 2 pre-integration: `hashio=df56cc69701bba1f5d395dba8b4f4289fe8e6d409fae21b01a4daaacd2c0db85`, `receipts=3264cacb8820a422ca8e55b18ea4819b8cb1de111edc4ab77c9842c583ede537`.
4. Torn integration: `types=6610a7f5226e341ff61a90bb89fbdbb75e375d11816ea14a0f8670ad090007f3`, `hashio=df872dc5709cc8814b5d11f87d4de8cc0351459cf5cdf04e19afd0a2baf2b13c`, `receipts=7cddf857a7b5f0fa9c420d94f13f80f91c0734ee3bf98c189e1639fa8a85bf9a`, seven-file root `910f4f9085ca1a643f60912fa8f6a2284dc8dd24fa43af0d521b43793b1b8d4c`.
5. Final frozen integration: the hashes in the table and seven-file root `755743d16190b9d055b0f6a3c341ada4b8a6084e0c49477287bddc2755ad50bd`.
6. Post-review canonicalization observed during persistence validation: `contract=b71255a35cce1701d2741064064b53ba819b742873a348f1852f147149e36467 -> 5ab8fb62dc558ba78e95fc07e52c1f1a1afbff5b1c5003834079eea6143d745f`, plan `5876b40075b488866960ab2c74f8aac5f655ef8df8727cdebdf89cc1d4814263 -> 4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e`, tracker `176b0ef18fe0b19ebdbe140638accb2d71fe3742be14a233042de54cd3a5ee71 -> 714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b`. This is a supersession record, not a Round 2 review binding.

No torn digest is used as a Round 2 authority binding.

## Resolved Round 1 findings

### Historical basis versus fresh candidate — resolved

The JSON now separates `historical_audit_basis` from `candidate_conformance_index`. The historical object canonical digest independently recomputes to `8fead557cfc236fe35468e39f69ceed162b5496fb8195a73587b31376cd2f1fb`. The candidate index contains no self hash, permits fresh member hashes to differ from history, and is externally named by a candidate-review receipt. P2 explicitly treats historical hashes only as provenance and does not self-lock a corrected implementation to them.

### K7 source-order rule — resolved declaratively

The amendment requires all candidate, review, runner, process, observation, runtime, reference, inner-summary, and closure checks before `OneUseEvaluator` construction or any consumed-state transition. That is the correct normative order. The frozen current `evaluator.py` still sets `_consumed=True` at lines 255–257 before validation; the amendment accurately labels it nonconforming, so no current executable candidate is accepted. The remaining runner-evidence blocker below also prevents external proof of the temporal property.

## Independent fixture replay

The embedded reference and a separately written scratch scorer both reproduced the complete fixture without importing production evaluator code.

### Canonical artifacts

| Artifact | SHA-256 |
|---|---|
| fixture input | `1264779925401d8efeecddefbe7245eca892543e923ec7c19ade08286fa062c9` |
| materialized 288 rows | `759b344fbc7db6ea78b041c84e23ff0fc59733985775be18f6d5c235e8408dbf` |
| component-order array | `8583d714619e7b1bddc755bac95b91a801a8c97951e382fbfd7f875a70f066f2` |
| normal expected output | `50f8376d0ee1e245a8521c7831d6d23635ac8cc74694ceeef1f028c8c48e6622` |
| attack expected-global-fail output | `2e62a7b830f453e0505cdeabb766f8a46c0ab2523321d3b1f933bf7fd2cc5f30` |
| reference source, 11571 ASCII bytes, one LF | `b23bf344a10f84a583c38e2326efd5ca70d4c4462fbf76715cd32b9afd20922f` |
| dependency contract | `646ac92f9b20eafb1ca96d8d06692984d4fca21b48a85de5f19ac2fa8c53576d` |
| reference-runtime contract | `c8bb58ec5d347573180d39ff369f32604737cccf8784da4cca47067c137d2d23` |

### F20/F21 point bytes

| Arm/condition | AvgNAE `<f8` LE hex | AvgOBO `<f8` LE hex |
|---|---|---|
| local / clean | `efeeeeeeeeeeae3f` | `888888888888e83f` |
| local / drift | `abaaaaaaaaaac63f` | `bcbbbbbbbbbbdb3f` |
| global / clean | `676666666666c23f` | `abaaaaaaaaaada3f` |
| global / drift | `676666666666d43f` | `111111111111d13f` |
| uniform / clean | `676666666666c23f` | `efeeeeeeeeeede3f` |
| uniform / drift | `cdccccccccccd23f` | `111111111111d13f` |
| capacity-control / clean | `555555555555c13f` | `111111111111e13f` |
| capacity-control / drift | `deddddddddddd33f` | `555555555555c53f` |

The strongest comparator is `uniform`; route margin is `deddddddddddbd3f` and paired clean delta is `565555555555b5bf`.

### Ordered component bootstrap

- component order: `cmp-07, cmp-00, cmp-05, cmp-02, cmp-06, cmp-01, cmp-04, cmp-03`
- PCG64 seed: `3783362605739069799`
- all 10,000 ordered `<i8[8]>` draw indices: `bcfccfc96c2781162e3c973dea8f88478b1912553acc730aa52772947ed8aef1`
- route raw `<f8[10000]>`: `8a325a51517a7a7669f4f8d4bdfa4d103204383b5d769e19cad409aba5f9e132`
- clean raw `<f8[10000]>`: `91d1fa29d36ad2e4e41733141d841a756c986b3670617f651315026a7157bdae`
- draw-pair root: `49b83785c436d2abd5cbcda617d142922f27ee21a4e17a39f730135add15cf3e`
- comparator counts: global `3567`, uniform `4133`, capacity-control `2300`
- route lower/upper: `2c9b6cb2c926abbf` / `e7a28b2ebae8c23f`
- clean upper: `692fa1bd84f6a2bf`

The exact locked Linux CPython 3.12.4 / NumPy 2.1.0 runtime was unavailable locally. Non-authoritative reproducibility checks on Windows CPython 3.12.0 / NumPy 1.26.4 and CPython 3.14.5 / NumPy 2.4.6 were byte-identical to the frozen results.

### Seventeen attacks

The independent validator returned, in order:

`A00 P05M_EMPTY_POPULATION`; `A01 P05M_INCOMPLETE_CROSS_PRODUCT`; `A02 P05M_DUPLICATE_OBSERVATION`; `A03/A04 P05M_ROW_KEYS`; `A05/A06 P05M_INTEGER_TYPE`; `A07/A08 P05M_NONFINITE`; `A09 P05M_POSITIVE_GT`; `A10 P05M_NONNEGATIVE_ESTIMATE`; `A11/A12 P05M_COMPONENT_ORDER`; `A13 P05M_IDENTITY_METADATA`; `A14 P05M_SEED`; `A15 P05M_TOKEN`; `A16 P05M_MIN_COMPONENTS`. Every result had exact `status=GLOBAL_FAIL`, and the complete result root equaled `2e62a7b830f453e0505cdeabb766f8a46c0ab2523321d3b1f933bf7fd2cc5f30`.

### Current production and inner-summary witness

Current production still derives lexical component order. It produced normal-output root `8674dca2cc84d5c0c538e966be7aade6a95bc809b3a56ebdf40b194a450f5d82`, route/clean raw hashes `9d13330237fed79f53a93db19edee3d7a66049333f269936afa325712ee507cd` / `79afebeb7a9b2bd3ba23b45432483eebc6846cd34c3d7c0cced6c37cf19b26b9`, draw-pair root `7b7182eae35fb32e84c7dcae874532cdf5abd692bd2c721521ab68b587f87e03`, and counts `3636/4145/2219`.

Despite that mismatch, passing the frozen input/expected/oracle bytes and current seven-file digest to `count_metric_fixture_receipt` produced a syntactically valid `status=PASS` inner receipt with SHA-256 `009cdfc196a484bc4f6b69f92ee337753345985f1dcf2d9d2b5a9dc3ba392736`. This is acceptable only if every consuming document and implementation enforces the outer runner receipt; they currently do not.

## Blocking findings

### R2-B1 — the declared seven-file closure is not the Python executable closure

A clean standard `import pams.temporac.evaluator` executed ten local files. In addition to the seven listed members it executed:

| Omitted executable file | SHA-256 |
|---|---|
| `src/pams/__init__.py` | `8b6ffbf3ac8ed7e43f1f40ef5793baa087f73b303eff342d891380bc9a5d67ce` |
| `src/pams/types.py` | `78c2ad53e98856425492a1711faa62fc2bb7764767dc48a22f7c4d5e043a3e31` |
| `src/pams/temporac/__init__.py` | `e3f71c2d85974ac3f635f76ee61f8e1f793b9bc251b25ff4336563d4da52c2a3` |

`src/pams/__init__.py` actively imports `pams.types`; changing any omitted initializer/dependency can change import-time behavior without changing the seven-file digest. The command/process schemas do not bind `sys.path`, resolved module origins, or an isolated loader that bypasses package initializers. The import allowlist also does not say whether it constrains direct source imports or the complete transitive NumPy/stdlib import graph. `runtime_lock_sha256` has descriptive semantics but no closed runtime-lock object/preimage in this amendment.

Separately, `gates.py` and `fixtures.py` enforce P2 and the fixture field names but are bound by neither the seven-file digest nor a runner closure. This is material: current `GateMachine.pass_p2` accepts arbitrary syntactically valid digests, and current `fixtures.py` retains the forbidden `attack_global_fail_sha256` field.

Required correction: bind the actual import/bootstrap files and module origins, define the allowlist scope and closed runtime-lock bytes, and bind all P2 gate/fixture enforcement either in the evaluator closure or in a separately complete runner/orchestration closure.

### R2-B2 — runner and observation evidence is not process-bound

`runner_code_sha256` is a bare 64-hex field. No runner source path, ordered member list, raw-byte preimage, dependency closure, command specification, executable, environment, process identity, or invocation-count witness defines what that hash covers. `reviewer_witness_sha256` is likewise named without a closed witness schema or verifiable independence semantics.

The process witness and observation root are sibling hashes rather than a linked execution record. The process witness binds stdout/stderr and command/runtime data but not an observation root. The observation root binds normal/attack/row/draw hashes but not command role, candidate index, evaluator closure, runtime, process identity, or process-witness hash. The amendment never requires stdout to be the canonical observation-root bytes. A clean no-op process can therefore be witnessed while an unbound runner copies frozen expected hashes into fabricated production/reference roots; the two role roots can also be exchanged without violating either closed schema.

Required correction: freeze a runner executable/dependency preimage and runner-process witness; make each production/reference observation an exact output of its named process by binding role, candidate, command, runtime, closure, process identity and observation bytes in one directionally closed object; and make the outer runner evidence prove exactly one fresh execution of each role and all 17 observed attacks.

### R2-B3 — plan/tracker/gate consumers still authorize the six-key inner receipt

In the reviewed plan snapshot, P2-METRIC emits `temporac.count-metric-fixture-receipt.v1` at line 112, S0 carries that receipt at line 145, the receipt table names it at line 162, and G5a/K7 consume it at lines 165–166 and 213. The reviewed tracker repeats the same chain at lines 47, 54, 73, 76, and 153. Neither document requires the candidate-review receipt, runner-evidence receipt, command/process witnesses, or production/reference observation roots introduced by Amendment 003.

The executable enforcement agrees with the old documents: `gates.py:112-123` marks P2 from two arbitrary digest strings, and `fixtures.py:412-459` models the old five-hash index with the forbidden attack field. Thus the document set still admits the concrete `009cdf…` self-issued PASS witness above even though the amendment locally says the six-key receipt is insufficient.

Required correction: atomically update proposal consumers, plan, tracker, gates, fixtures, S0/G5a/K7 surrounding indexes, and tests so only a verified outer runner-evidence digest can pass P2 and flow downstream. The six-key receipt may be retained only as a bound inner summary.

### R2-B4 — inner-summary presence is both optional and mandatory

Amendment JSON `cross_stage_verification.S0` and Markdown line 527 call the inner-summary receipt optional. JSON `K7_prestart` and Markdown line 538 require its exact bytes from S0, while the runner receipt closed keys always require `inner_summary_receipt_sha256`. Therefore an S0 artifact can conform by omitting the receipt and must later fail K7 for omitting it.

Required correction: choose one rule. If the runner receipt retains a non-null inner-summary digest, make the exact inner-summary bytes and digest mandatory at P2/S0 and required only as a non-authoritative runner input. Otherwise remove the field and every K7 requirement for it. Do not leave a conforming S0 state that is guaranteed `STOP-METRIC`.

## Non-expansion and authority ceiling

The amendment preserves F20/F21 signs and reductions, the ordered three seeds, 10,000 PCG64 draws, comparator tie order and paired clean reuse, one claim, four blocks, 27 training jobs, 93 allocated A6000 hours, all scientific thresholds, data boundaries, and F23 order. It adds no natural row, source identity, model, job, GPU work, result, claim, or paper-visible block.

The Round 2 result is therefore `REVISE`, not rejection of the fixture mathematics. Acceptance requires closing all four evidence/consumer blockers and a fresh review of the newly versioned amendment and its complete executable/orchestration closure. Same-family provisional is the maximum status; this review authorizes nothing.
