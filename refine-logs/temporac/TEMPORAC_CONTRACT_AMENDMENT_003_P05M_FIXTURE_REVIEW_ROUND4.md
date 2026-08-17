# TempoRAC Contract Amendment 003 P05M Fixture — Round 4 Normative Review

Date: 2026-08-16  
Reviewer: `/root/temporac_p05m_amendment_review`  
Route: Codex `gpt-5.6-sol`, OpenAI same-family continuation  
Assurance: provisional; not cross-family independent  
Verdict: **ACCEPT**  
Blocking groups: **0**  
Authority: **none**

## Outcome

Round 4 closes both Round 3 blockers and introduces no replacement blocker.
The amendment now specifies an executable, noncircular evidence contract for
the future P05M candidate: all four roles use a dedicated anonymous FD-3
attestation channel; the parent observes EOF before stdout; actual post-scoring
module, native-image, sandbox, process, and empty-mount state is carried in the
closed witness chain; and the PASS writer plus non-PASS capture routine are in
the exact future 18-member candidate/P2-verifier closure.

The amendment is accepted as a normative specification only. The runner,
candidate index, runtime lock, command-template hashes, process observations,
gate evidence, and capture bundle are intentionally not materialized. The
current 17-member implementation is explicitly recorded as historical and
nonconforming. Consequently this ACCEPT does not pass P2, S0, G5a, or K7 and
does not authorize implementation, execution, data access, training, launch,
server work, Git changes, results, paper promotion, or claims.

This is the same reviewer and same model family as earlier rounds. The maximum
finding is therefore **same-family provisional acceptance of the specification
for a later, separately materialized and freshly reviewed implementation**.

## Reviewed bytes and integrity

The fixed aliases and timestamped `_20260816_164801` files were compared
byte-for-byte and are identical.

| Direct input | Bytes | SHA-256 |
|---|---:|---|
| timestamped amendment Markdown | 40,558 | `a6b644aaaa29cf6c0423cdd3b287b183a88a9f23cbee3a04afc6db932a4d9005` |
| fixed amendment Markdown alias | 40,558 | `a6b644aaaa29cf6c0423cdd3b287b183a88a9f23cbee3a04afc6db932a4d9005` |
| timestamped amendment JSON | 114,825 | `14d59552242965a717f1162208770cb8ded9ebce9f7e060f904ba327617a052a` |
| fixed amendment JSON alias | 114,825 | `14d59552242965a717f1162208770cb8ded9ebce9f7e060f904ba327617a052a` |
| Round 3 review Markdown | 23,288 | `7055c41896a728d7612e352b390d8e8a02c040a1bfe086a8a51dc162a3e734b7` |
| Round 3 review JSON | 22,469 | `35d8640283a75efef7c0b4fe6a53f7cba803ceb3d49dc58dd7f680e0102700be` |
| canonical proposal, both aliases | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| experiment plan | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| experiment tracker | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |
| research contract | 12,060 | `3e037a2f74c49dd62e596dc7b875d94216e45b7f693b80a7afb6a60f8244367b` |

The amendment JSON decodes as strict UTF-8, has no BOM or CR, exactly one
terminal LF, no duplicate key, nonfinite number, or negative zero. The
Markdown reference block is exactly 11,571 ASCII/LF bytes with one terminal LF.
All declared canonical subobject commitments recompute exactly.

Applicable current protocol mirrors were also fully reread:

| Protocol | Bytes | SHA-256 |
|---|---:|---|
| `docs/BASELINE_PROTOCOLS.md` | 7,623 | `2084578b70797bf2988768848b96b23f20013d92ff32d13334166350d19d910d` |
| `.agents/skills/research-refine/SKILL.md` | 30,984 | `bb489173329fec7a6551c74c319e2103d83f722ea11fc11462a7c4ccacd00b09` |
| `output-versioning.md` | 4,828 | `de8e7ac23069de6c5d0c482d4cef117dd5a543e00b57e2dc6c62560a457290c5` |
| `output-manifest.md` | 1,753 | `e767083e0ef97a7adf7a4e8a574090067a59c0655d5aca422a3747cd32b6afde` |
| `output-language.md` | 2,215 | `0f1447579bd7a2195fd4a074be70ae6345b5c368f1a70d327e89748bf5a378f9` |
| `output-composition.md` | 5,293 | `79a72c7ea02102c460ae3ad55853c21a4054b9df93f377c29dd34555e6d3c3a7` |

The five `skill://` rows inside `historical_audit_basis` are immutable
historical provenance, not live candidate inputs. The complete historical
basis canonical root is
`9f2c09addedd942e5781a0823c2b359ce3872bdcb12df62c84d2325316f2fb03`.
All 60 filesystem-backed members matched their declared byte counts and hashes.

## Round 3 blocker closure matrix

| Round 3 item | Round 4 result | Independent conclusion |
|---|---|---|
| R3-B1: actual process state lacked a closed transport and work isolation | **CLOSED** | P2 verifier, controller, production, and reference each write one canonical post-scoring attestation to child FD 3 over a distinct `pipe2(O_CLOEXEC)` pipe. The immediate parent retains the fixed read slot, validates the pipe token, drains through EOF before stdout, and derives the process witness from received bytes, pidfd exit, argv/environment, and a retained mount handle. Actual complete `sys.modules` origins, `/proc/self/maps` plus recursive ELF closure, sandbox state, FD table, and mount identity are embedded rather than substituted by an expected allowlist. Four pairwise-distinct initially/finally empty read-only mounts eliminate the former shared writable side channel. |
| R3-B2: PASS-writing P2 verifier had only a bare digest | **CLOSED** | The P2 verifier closure is exactly the 17 production members followed by `tests/temporac/test_p05m_fixture.py`, using the candidate-snapshot prefix and raw path/length/member preimage. Candidate review, P2, S0, and K7 require elementwise member equality and `p2_verifier_closure_sha256 == candidate_snapshot_sha256 == SHA256(exact 18-member preimage)`. Both `_emit_gate_evidence_pass` and `_capture_p2_gate_bundle` are entry points in that runner member. No independent 64-hex closure value is accepted. |

### FD and mount executable consistency

The descriptor rules are executable rather than contradictory. Each parent
creates the attestation pipe while its pre-pipe FD set is exactly `[0,1,2]`
for the outer capture root or `[0,1,2,3]` for an attested parent. It may then
allocate stdin/stdout/stderr, pidfd, and retained mount handles. Spawn file
actions map only the child write end to FD 3 and close all unrelated child
descriptors. Those later parent handles are closed before the parent's own
post-scoring FD snapshot. Production is fully closed before reference reuses
controller read FD 4.

The four empty mounts can be created by the outer sandbox launcher before the
corresponding role enters its witnessed deny-mutation lifetime. The role sees
only its own already-created read-only mount; the parent retains a handle and
rechecks identity and final emptiness after exit. Thus mount creation does not
conflict with the role-lifetime filesystem-mutation denial.

The channel preimage binds role, candidate index, challenge, attestation nonce,
independently observed pipe token, parent/child PIDs, Linux start ticks, child
FD 3, and the fixed parent read slot. Nonce/challenge reuse, sibling-pipe swaps,
stdout-before-EOF, extra bytes, alternate transports, extra FDs, changed
mounts, late imports, or copied expected manifests fail closed.

### Noncircular gate evidence

The production and reference workers attest the exact observation bytes before
stdout. The controller embeds both observations and both child process
witnesses in non-PASS runner evidence, then attests its own exact output. The
P2 verifier validates the exact 18-member equality, fresh candidate review,
inner six-key summary, runner evidence, and every nested process/scientific
equality before writing nested `gate-evidence.v2` PASS.

That nested PASS cannot contain its own later exit witness and is explicitly
inert alone. Only after FD-3 EOF, exact stdout, empty stderr, pidfd exit zero,
and final mount validation does the outer capture routine construct the
six-key `p05m-gate-capture-bundle.v1` with status `CAPTURED_COMPLETE`. The
capture routine is itself in the reviewed runner member, while the existing P2
stage observation root contributes no PASS semantics. This direction is not a
self-certification cycle.

The legacy six-key `temporac.count-metric-fixture-receipt.v1` remains mandatory
subordinate evidence but is `NEVER_SUFFICIENT_FOR_ANY_GATE`. The sole input at
every named P2/S0/G5a/K7 plan/tracker interface is the complete capture bundle,
under semantic-overlay root
`48770609fe6ecbbe8ee456b9c046ab5fb5e8f87e06d1fbe00fc1771b3f7c7e90`.

K7 recursively verifies the bundle, fresh review, immutable candidate, 17/18
member closures, DAG, runtime/native trees, four commands, transport, mounts,
processes, observations, and S0/G5a/grant equality before constructing or
starting the evaluator. Mismatch consumes nothing; successful OS creation of
the sole evaluator is the only consumption point.

## Closure and dependency witnesses

The historical 17-member production preimage was independently rebuilt from
raw files. It is 316,199 bytes and hashes to
`dfcf258b13aa571d3636101f09e66c9d7889a0f09dd88515f4c332ae2d662943`.
Independent AST extraction found exactly the declared 17 module origins and 34
unique sorted local-import edges, no dynamic import/exec/eval edge creator,
and an acyclic graph. The canonical DAG root is
`e3a6170a5d54a3bd9373939351caabeb70cbaeef5695c5aa26b3de94bb261073`.

The future 18th runner member is absent, as the amendment requires at this
stage. Therefore no authoritative candidate/P2 root was invented or accepted.
The amendment fixes its only legal member order, prefix, preimage, equality,
entry points, runtime binding, and fresh-review condition; materialization and
fresh implementation review remain separate future work.

| Canonical contract object | SHA-256 |
|---|---|
| historical audit basis | `9f2c09addedd942e5781a0823c2b359ce3872bdcb12df62c84d2325316f2fb03` |
| plan/tracker semantic rebinding | `48770609fe6ecbbe8ee456b9c046ab5fb5e8f87e06d1fbe00fc1771b3f7c7e90` |
| production dependency DAG | `e3a6170a5d54a3bd9373939351caabeb70cbaeef5695c5aa26b3de94bb261073` |
| evaluator dependency contract | `1f1747012f047d8db69944391f3a63e11ada17edc163697608d3aed6f2758f6c` |
| reference runtime contract | `3343fdcf99eba48c3c21ad2b65008855cd398cdfa04ed66d7a9ee634e9a32a3d` |
| runtime-lock contract | `bff68eb3db94c4bf5e56e5ba8e2d6f1e961305f5752e6ce748d295f70eb2ba0a` |
| sandbox-profile contract | `1c6be7154c6b8a5727fc1bc22e83b6796855614990a1c778819d9e6c17be5c2a` |
| attestation-transport contract | `f01223582fe6ac29112ead3cf172f6ea04664a84a202b30668a92774a238ba47` |
| gate-capture-bundle contract | `46bb12eba46483b4846db41bcd67f437cc4965e74a94890f132b4aa14a8d9bcf` |
| P2-verifier closure contract | `d297372d44e60e7c2443c6295f99ef6cfed3afe7e21b4d993ad81e9cdd7396d9` |
| historical 17-member production closure | `dfcf258b13aa571d3636101f09e66c9d7889a0f09dd88515f4c332ae2d662943` |
| reference source | `b23bf344a10f84a583c38e2326efd5ca70d4c4462fbf76715cd32b9afd20922f` |

## Independent fixture and science witnesses

The exact reference block was compiled in memory with filename
`<temporac-p05m-reference-v1>`. A separate reduction replay and the current
production F20/F21 point implementation agreed byte-for-byte. The diagnostic
host was Windows CPython 3.12.13 with NumPy 1.26.4, not the future locked Linux
CPython 3.12.4/NumPy 2.1.0 runtime, so this replay is evidence for deterministic
specification arithmetic, not runtime or execution authority.

| Witness | Independent result |
|---|---|
| fixture input canonical bytes/root | `6,191` / `1264779925401d8efeecddefbe7245eca892543e923ec7c19ade08286fa062c9` |
| row count / canonical bytes/root | `288` / `43,032` / `759b344fbc7db6ea78b041c84e23ff0fc59733985775be18f6d5c235e8408dbf` |
| component-order bytes/root | `74` / `8583d714619e7b1bddc755bac95b91a801a8c97951e382fbfd7f875a70f066f2` |
| normal expected bytes/root | `1,705` / `50f8376d0ee1e245a8521c7831d6d23635ac8cc74694ceeef1f028c8c48e6622` |
| 17-attack expected bytes/root | `1,711` / `2e62a7b830f453e0505cdeabb766f8a46c0ab2523321d3b1f933bf7fd2cc5f30` |
| PCG64 seed / draws / components | `3783362605739069799` / `10000` / `8` |
| raw sampled-index stream, 640,000 bytes | `bcfccfc96c2781162e3c973dea8f88478b1912553acc730aa52772947ed8aef1` |
| route raw `<f8[10000]`, 80,000 bytes | `8a325a51517a7a7669f4f8d4bdfa4d103204383b5d769e19cad409aba5f9e132` |
| clean raw `<f8[10000]`, 80,000 bytes | `91d1fa29d36ad2e4e41733141d841a756c986b3670617f651315026a7157bdae` |
| draw-pair root | `49b83785c436d2abd5cbcda617d142922f27ee21a4e17a39f730135add15cf3e` |
| comparator draw counts | global `3567`, uniform `4133`, capacity-control `2300` |
| strongest comparator | `uniform` |
| F21 route margin / clean delta | `deddddddddddbd3f` / `565555555555b5bf` |
| route lower/upper; clean upper | `2c9b6cb2c926abbf` / `e7a28b2ebae8c23f`; `692fa1bd84f6a2bf` |

F20 binary64 little-endian witnesses:

| Metric | local clean/drift | global clean/drift | uniform clean/drift | capacity clean/drift |
|---|---|---|---|---|
| MAE | `efeeeeeeeeeeae3f` / `abaaaaaaaaaac63f` | `676666666666c23f` / `676666666666d43f` | `676666666666c23f` / `cdccccccccccd23f` | `555555555555c13f` / `deddddddddddd33f` |
| OBO | `888888888888e83f` / `bcbbbbbbbbbbdb3f` | `abaaaaaaaaaada3f` / `111111111111d13f` | `efeeeeeeeeeede3f` / `111111111111d13f` | `111111111111e13f` / `555555555555c53f` |

All attacks returned only the frozen global error and no partial score:

`A00 P05M_EMPTY_POPULATION`; `A01 P05M_INCOMPLETE_CROSS_PRODUCT`;
`A02 P05M_DUPLICATE_OBSERVATION`; `A03/A04 P05M_ROW_KEYS`;
`A05/A06 P05M_INTEGER_TYPE`; `A07/A08 P05M_NONFINITE`;
`A09 P05M_POSITIVE_GT`; `A10 P05M_NONNEGATIVE_ESTIMATE`;
`A11/A12 P05M_COMPONENT_ORDER`; `A13 P05M_IDENTITY_METADATA`;
`A14 P05M_SEED`; `A15 P05M_TOKEN`; `A16 P05M_MIN_COMPONENTS`.

Normal `status=PASS` is correctly defined as scorer conformance only, not a K7
scientific result. The synthetic draw lower bound is negative; that fact is
preserved and is not promoted into a scientific claim.

## Complete filesystem-backed historical input manifest

These are fresh local rehashes, not acceptance of copied author values.

### Planning, archives, and prior reviews

| Path | Bytes | SHA-256 |
|---|---:|---|
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE_20260816_145644.md` | 29,722 | `d615def9dc468b7e483488b87cb0524302f1c4c325df8bf092e741e21f6262d1` |
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE_20260816_145644.json` | 29,094 | `c73b85731ece838684f78c41b3dfcdb498af5cf9822f7ee84d2add4d09ffe1fc` |
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE_REVIEW_ROUND1.md` | 6,961 | `ca83e5277dfcd9319d331ef5fbb986fb0e37746c27181fb7f31cfb82088b42e4` |
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE_REVIEW_ROUND1.json` | 7,033 | `2df5d40dc87c8e829cffd531f1cd89d8666df24a93fdca59e19f51b0b34dac4d` |
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE_REVIEW_ROUND2.md` | 17,015 | `9f4dc98b3f2237dbd17381007105977639b13d60b478e13e162631666504296b` |
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE_REVIEW_ROUND2.json` | 12,771 | `b8552d10bf8b9d9e6b3f9202127b1ea56f9b3ea9d43a9f9994e06166ce741fe2` |
| `refine-logs/temporac/FINAL_PROPOSAL.md` | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| `refine-logs/EXPERIMENT_PLAN.md` | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| `refine-logs/EXPERIMENT_TRACKER.md` | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |
| `idea-stage/docs/research_contract.md` | 12,060 | `3e037a2f74c49dd62e596dc7b875d94216e45b7f693b80a7afb6a60f8244367b` |
| `TEMPORAC_CERTIFICATE_IMPLEMENTATION_REVIEW_POSTFIX_ROUND2_20260816.md` | 12,283 | `93a9cae771b34c185b9057286794b894d615aec5424011df145aaf8900f008d8` |
| `TEMPORAC_CERTIFICATE_IMPLEMENTATION_REVIEW_POSTFIX_ROUND2_20260816.json` | 14,527 | `144d2f22f4f75166b8cbefe54157d08690408979dd7cb34ef8c97e1738432872` |
| `TEMPORAC_OPERATOR_IMPLEMENTATION_REVIEW_V2_20260816.md` | 15,244 | `73c660079377683b2066e4e04e8b6400f3568e1b7473069303908f773d0caad1` |
| `TEMPORAC_OPERATOR_IMPLEMENTATION_REVIEW_V2_20260816.json` | 14,108 | `25792a855e15769ec778276b5066052d15796eea18cad5d44a16f741cb8bd35f` |
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE_20260816_162320.md` | 45,024 | `7ed7dd65adba298a176cfef52ea6bc5464a664ec7d664ef49995ec154895f71b` |
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE_20260816_162320.json` | 81,019 | `09b7a835bcc3e3d3c5193b23830c1afc97811665953aaefffc867d8abcf619c7` |
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE_REVIEW_ROUND3.md` | 23,288 | `7055c41896a728d7612e352b390d8e8a02c040a1bfe086a8a51dc162a3e734b7` |
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE_REVIEW_ROUND3.json` | 22,469 | `35d8640283a75efef7c0b4fe6a53f7cba803ceb3d49dc58dd7f680e0102700be` |

### Current 17-member production closure

| Path | Bytes | SHA-256 |
|---|---:|---|
| `src/pams/__init__.py` | 160 | `8b6ffbf3ac8ed7e43f1f40ef5793baa087f73b303eff342d891380bc9a5d67ce` |
| `src/pams/types.py` | 6,210 | `78c2ad53e98856425492a1711faa62fc2bb7764767dc48a22f7c4d5e043a3e31` |
| `src/pams/temporac/__init__.py` | 280 | `e3f71c2d85974ac3f635f76ee61f8e1f793b9bc251b25ff4336563d4da52c2a3` |
| `src/pams/temporac/contract.py` | 9,079 | `5ab8fb62dc558ba78e95fc07e52c1f1a1afbff5b1c5003834079eea6143d745f` |
| `src/pams/temporac/types.py` | 25,746 | `4249eac2738f5b2edf53f38cbb681dffae609c151d493dd03b625fe10ad4c3ca` |
| `src/pams/temporac/hashio.py` | 27,198 | `aa460d5de51f107b2fc6b2ad171747e0da38a87cf30f82a869a335c268f1d613` |
| `src/pams/temporac/receipts.py` | 27,223 | `564039b5d37a4a2759544a9ae65f267b57d7f6c1e46bca812ae5ed4f4a2c17e4` |
| `src/pams/temporac/trusted_packer.py` | 17,137 | `40d6e3d5be1b26b7ba0f0f7aadf8da7496c95c4e829face5c7e9faaee2cdf94e` |
| `src/pams/temporac/quadrature.py` | 11,569 | `44f4b91dc4d5631442982732177a6f57726c0b8f42594380b5a75f2209efc2f3` |
| `src/pams/temporac/preprocess.py` | 24,916 | `042b5cae1917359cb5596837c81c0e36658ff99a6a8fd6f2602b478ca6f98bc3` |
| `src/pams/temporac/x0.py` | 24,963 | `5f4585425a83bf63779265db9004d210c5c2875a579ebc9a6c51297eb647c7a2` |
| `src/pams/temporac/certify.py` | 50,212 | `468ce8418bff0f43582d341f3e1b8064cf1c429301d1f56be84b8e13ad59a3f1` |
| `src/pams/temporac/metrics.py` | 17,608 | `1dadf08e89330bed2ab773afda05bffcb3fe2252b0d3e23c0bcc18c2c3b049dd` |
| `src/pams/temporac/evaluator.py` | 21,977 | `f981c6c7061c3388f06ef39466ef98192e65c62c501821be2b211f903cfaf9de` |
| `src/pams/temporac/decode.py` | 8,843 | `766cd6ca4b9e133041bd602f50fde840ec7caadd376bd6851799583d44f31f5e` |
| `src/pams/temporac/fixtures.py` | 36,365 | `979b8cee4fed117f5471cf399794f615c60eae52dbba2c946109ff836206b464` |
| `src/pams/temporac/gates.py` | 5,929 | `b6c9d7432443d7ad36738a81700746709fd571d475514aa6f1bbcd39be744b4a` |

### Current TempoRAC tests

| Path | Bytes | SHA-256 |
|---|---:|---|
| `tests/temporac/__init__.py` | 55 | `dd07192e3b73d582dfb5f0325cfafd7462040c76574ad8241a661fdc4713a24e` |
| `test_certificate_artifact.py` | 19,943 | `f2b0ef6e6d977bf3532595147fba89b611258063ee5e36e30657c31c62e56c20` |
| `test_certificate_tau.py` | 4,665 | `f7e2f25ab93226568910b856167e441037213ba3dd937d6915587037b7c6a94b` |
| `test_contract_hashio.py` | 6,272 | `524c9ddf04e8cd29735c2c7d958f5f9e31afd8fea0bb3b6068a09b2de75b7104` |
| `test_feature_io_firewall.py` | 5,707 | `f049e54c7254a848dc6cdd10afb853c5538fd8ecabd02b86950eb77d8a74c12b` |
| `test_gates_fixtures.py` | 6,691 | `d6d91a7dd606c42f672bb015d77c5c10406c2c6de69ed3a8bff76a0ce4f713d3` |
| `test_metrics_evaluator.py` | 13,066 | `a6f4aa275796b06162cac863674205d85b07adc46f89bc8d9821c48d17440fc6` |
| `test_nola_decode.py` | 4,632 | `80586866a9293e892c2cab1ca7318b97d341c2aa5990f7e1a31695f18376ff0b` |
| `test_objective_runtime.py` | 5,583 | `cc3685e2bf53ed7535addb1c09f08e0520dbd9143ba5d44da1cca28761107fa5` |
| `test_operator_manifest.py` | 24,090 | `7f2c90c320d7378a74fcc64f6c808eb6e426cdcfbd73cab0fb1242f690db9db0` |
| `test_prediction_identity.py` | 4,391 | `7967b11ee6585c2664fa4f39f5a9d9798ab407317f0d4efcf790f8d7a7bd5148` |
| `test_prediction_receipts.py` | 5,131 | `25fe8cc8696ab6c4108ba909b2f315c83b674fbe6ec9a770e6b7f6e449b410d4` |
| `test_preprocess_geometry.py` | 4,926 | `2edda5c593992c32aca30452340fce357930262978e331d37b6e5d55d3eb3b3f` |
| `test_quadrature_cue.py` | 4,362 | `4b7f977cbd31dc441ac021666bbc036b1d7de505ee884445df356b57b88374b6` |
| `test_response_invariants.py` | 8,307 | `b2e88e011bb06b9e21011ff252b630ffcbfcbe53a29638f8983b563f00ffae63` |
| `test_teacher_certificate.py` | 6,944 | `a5dc0bff236f417ab570193e46542b693d0f201fedd91693e6c572a70d20e012` |
| `test_training_fresh_graph.py` | 4,285 | `52b9f266173fec04152240d629dbf651aadd5b85d5839abcfe9f16fedc608845` |
| `test_x0.py` | 8,831 | `90206f53c7fffeabe9bcb3d6374fd60d3c51c69a7d1137df0cba797b583d1ef8` |
| `test_x0_manifest.py` | 7,123 | `074bb8df9dce5d1ea9e987499dac0d8824e1803a5f4bdc32495bd957e8c4c6a3` |

The remaining current `src/pams/temporac` modules were read as regression
context but are deliberately outside the P05M production closure:

| Module | Bytes | SHA-256 |
|---|---:|---|
| `cue.py` | 19,289 | `e702784bef27db94c1dc027d036a64d7461ae618613baec5e68061254f98569c7` |
| `feature_io.py` | 6,750 | `f54f17ca204c242d304314f4280e7540c28588e86ea588a405fff02176b95428` |
| `nola.py` | 14,395 | `ad6a13203f09c13a37a854333d4458ca8ce5a03bb186c0946f014a4e8cb9a637` |
| `objective.py` | 8,824 | `09a77231a48a869cbc83e0aa1aa2502f86f4abbcbab8c84f2b2c5abbd729ac87` |
| `prediction.py` | 7,988 | `7c57725240e83b7c06227a3b27e5f3cadb4590170488e5e126180caba8c13808` |
| `response.py` | 14,185 | `04df32d6e2e4061d86d813f611fb9ff277f2be8312af4efbc7219785af02f9bd` |
| `runtime.py` | 15,394 | `9421661046ee686508607529ed972b087e724c2c3a5c487e220c2f8e4038a0c8` |
| `teacher.py` | 12,157 | `30575fa64875aa6b77de157cf6af348b3b7900851a1c1742993842eea7928b30` |
| `training.py` | 10,375 | `0af8ecbadfcda43d2b9f60853c99210b316786ff4310ed5d0a32d9e366f627a4` |

Historical virtual provenance rows, reproduced only as declared immutable
metadata, are: research-refine skill
`2cbda50557603d608990eff7f7a9890b2f75978a5219a8bc8b6e49cebe7e1884`,
output-versioning
`73a4269b537be728b0ecc35789f11fa6d1fafa6d0430d07ad2738f064b1e4da3`,
output-manifest
`28e29bc8e53c52afe4eeeb590df8893282a3bbced08e105e154ad228b6cd1922`,
output-language
`7b326400dca5130ce4beedf72fc2010680afe96eb56a6ab21210d323adc9a52e`,
and output-composition
`8de81708431091a21a641be5f91db7958bb9f9a4b85d4fe03f52c5b496948651`.
They are not candidate locks and were not substituted for the current mirrors.

## Non-expansion, disposition, and authority ceiling

The amendment preserves F20/F21, the nonlexical component order, video-first
then ordered-seed reduction, 10,000 paired draws, comparator reselection,
clean-arm reuse, all scientific thresholds, 27 training jobs, 93 allocated
A6000 hours, one claim, four experiment blocks, and F23 order. It adds no
natural row, model, GPU work, training, result, claim, paper block, or resource
authority. Historical bytes remain provenance; the future candidate is bound
only by a separate candidate index and fresh review, so P2 does not self-lock.

Final disposition: **ACCEPT**, with zero blockers, for the normative Amendment
003 Round 4 bytes only. A future implementation must materialize and freshly
review the exact candidate index, runner, 18-member equality, runtime lock,
four command templates, process evidence, observations, and capture bundle.

All authority remains zero: `P0=0`, `P1=0`, `P2=0`, `P2-METRIC=0`, `P3=0`,
`S0=0`, `G5a=0`, `K7=0`, `gate=0`, `capability=0`, `evaluator=0`, `launch=0`,
`server=0`, `data=0`, `GPU=0`, `training=0`, `test=0`, `results=0`, `Git=0`,
and `paper/claim=0`. No amendment, code, test, proposal, plan, tracker,
research contract, MANIFEST, server, data, experiment, or Git state was
modified by this review.
