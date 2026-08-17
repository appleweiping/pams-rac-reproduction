# TempoRAC Contract Amendment 003 P05M Fixture — Round 3 Normative Review

Date: 2026-08-16  
Reviewer: `/root/temporac_p05m_amendment_review`  
Route: Codex `gpt-5.6-sol`, OpenAI same-family continuation  
Assurance: provisional; not cross-family independent  
Verdict: **REVISE**  
Blocking groups: **2**  
Authority: **none**

## Outcome

Round 3 closes the substance of Round 2 findings B1, B3, and B4. The fixed
17-member production closure, its 34-edge dependency DAG, package initialization
order, runtime/native contracts, runner source preimage, plan/tracker semantic
overlay, mandatory subordinate six-key receipt, and K7 verify-before-consume
order are all materially stronger and independently reproducible. The complete
288-row fixture, F20/F21 reductions, 10,000-draw PCG64 bootstrap, raw-f64 roots,
and all 17 attacks also reproduce exactly.

The amendment is nevertheless not yet an executable evidence contract. Two
authority-bearing gaps remain:

1. the frozen process I/O cannot carry or independently attest the post-scoring
   module/native/sandbox data required by the child and controller process
   witnesses, while the writable work mount is neither role-isolated nor closed
   against an unbound side channel; and
2. the PASS-writing P2 verifier has only a bare
   `p2_verifier_closure_sha256` field, with no members, preimage, equality rule,
   runtime/origin witness, or executable identity.

A verifier can therefore accept a fabricated process manifest or an arbitrary
P2-verifier digest while every explicitly frozen hash equality still holds.
The disposition is `REVISE`, not conditional acceptance.

This same-reviewer continuation is same-family provisional. It grants no
implementation, test, P2, S0, G5a, K7, gate, capability, evaluator, launch,
server, data, GPU, training, result, Git, paper, or claim authority.

## Reviewed bytes and integrity

The sole Round 3 amendment input was the timestamped `_20260816_151047` pair.
The fixed aliases were independently compared and are byte-identical.

| Input | Bytes | SHA-256 |
|---|---:|---|
| timestamped amendment Markdown | 45,024 | `7ed7dd65adba298a176cfef52ea6bc5464a664ec7d664ef49995ec154895f71b` |
| fixed amendment Markdown alias | 45,024 | `7ed7dd65adba298a176cfef52ea6bc5464a664ec7d664ef49995ec154895f71b` |
| timestamped amendment JSON | 81,019 | `09b7a835bcc3e3d3c5193b23830c1afc97811665953aaefffc867d8abcf619c7` |
| fixed amendment JSON alias | 81,019 | `09b7a835bcc3e3d3c5193b23830c1afc97811665953aaefffc867d8abcf619c7` |
| Round 2 review Markdown | 17,015 | `9f4dc98b3f2237dbd17381007105977639b13d60b478e13e162631666504296b` |
| Round 2 review JSON | 12,771 | `b8552d10bf8b9d9e6b3f9202127b1ea56f9b3ea9d43a9f9994e06166ce741fe2` |
| canonical proposal, both aliases | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| experiment plan | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| experiment tracker | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |
| research contract | 12,060 | `3e037a2f74c49dd62e596dc7b875d94216e45b7f693b80a7afb6a60f8244367b` |

The amendment JSON is strict UTF-8 without BOM or CR, has exactly one terminal
LF, contains no duplicate keys, non-finite number, or negative zero, and its
keys are recursively UTF-8 lexical. The three canonical subobjects and every
declared contract digest recompute exactly. All 50 filesystem-backed members of
`historical_audit_basis` matched their declared byte counts and SHA-256 values
at initial and pre-persistence rehash. The five `skill://` rows remain frozen
historical provenance, not candidate acceptance inputs; current project-local
protocol mirrors were read separately and are listed below.

| Applicable current protocol | Bytes | SHA-256 |
|---|---:|---|
| `docs/BASELINE_PROTOCOLS.md` | 7,623 | `2084578b70797bf2988768848b96b23f20013d92ff32d13334166350d19d910d` |
| `.agents/skills/research-refine/SKILL.md` | 30,984 | `bb489173329fec7a6551c74c319e2103d83f722ea11fc11462a7c4ccacd00b09` |
| `output-versioning.md` | 4,828 | `de8e7ac23069de6c5d0c482d4cef117dd5a543e00b57e2dc6c62560a457290c5` |
| `output-manifest.md` | 1,753 | `e767083e0ef97a7adf7a4e8a574090067a59c0655d5aca422a3747cd32b6afde` |
| `output-language.md` | 2,215 | `0f1447579bd7a2195fd4a074be70ae6345b5c368f1a70d327e89748bf5a378f9` |
| `output-composition.md` | 5,293 | `79a72c7ea02102c460ae3ad55853c21a4054b9df93f377c29dd34555e6d3c3a7` |

The explicit task restriction against editing `MANIFEST.md` controls this
review; no manifest entry was written.

## Blocking finding R3-B1 — actual process witnesses have no closed evidence transport, and the work mount permits an unbound channel

The amendment requires every `temporac.p05m-process-witness.v1` to embed the
complete post-scoring `module_origin_manifest`, its digest, a native-image
manifest digest, and an embedded sandbox witness (JSON lines 1258-1290). Those
values are properties of the observed child after scoring, not merely expected
runtime-lock values.

The only successful child output is, however, exactly one closed
`temporac.p05m-role-observation.v1` plus LF (JSON lines 1209-1221 and
1388-1402). Its nine keys contain no module-origin manifest, native-image
manifest, sandbox witness, or process witness. Successful stderr is empty and
stdin contains only the closed worker request. The controller construction rule
then says it verifies those absent observations in each child witness (JSON
lines 1431-1437), but no extra file descriptor, pipe, sidecar, tracer,
attestation record, or canonical transport is defined.

The same gap recurs one level outward. The controller stdout is exactly the
closed runner-evidence object; that object embeds child witnesses but no
controller witness (JSON lines 1405-1442). Only after controller exit does P2
construct the controller witness (JSON lines 1463-1466). An OS process handle
can confirm pid/exit status; it does not reveal the exited interpreter's
complete post-scoring `sys.modules` dictionary. With no frozen monitor or
in-process attestation bytes, P2 cannot distinguish an actual complete manifest
from a copied expected allowlist.

The work-directory rule makes this worse. Markdown line 520 calls the working
root read-only, while the JSON-authoritative `work_mount` says
`$WORK/empty` is writable (JSON line 987). The JSON precedence resolves the
wording in favor of writable, but it does not say whether controller,
production, and reference receive three distinct mount identities or one
shared mount. All three command templates and witnesses use the same
`$WORK/empty` token, and the witness binds only a constant initial-empty digest,
not a role-specific mount identity, final state, or write prohibition. A
conforming implementation may therefore pass manifests through an unbound file
or allow production/reference cross-role communication; a verifier cannot tell
that execution from an isolated one.

This is a concrete replacement attack: emit the correct closed observation on
stdout, populate the required witness from expected locked rows rather than the
actual child state, and use an unspecified extra FD or shared work file if any
coordination is needed. Every currently stated stdout/root/hash equality can
still pass.

Minimal repair:

- define one exact parent-observable attestation transport for every role,
  including descriptor/path identity, creator, bytes, schema, ordering,
  lifetime, and hash preimage; include any monitor/launcher in a reviewed
  executable/runtime closure;
- place the complete post-scoring module-origin and native-image observations
  on that frozen chain, and specify how completeness is verified rather than
  copied from the allowlist;
- give controller attestation the same treatment before exit; and
- make work roots explicitly read-only, or bind three distinct per-role mount
  identities and prohibit cross-role access, with initial and final tree
  witnesses. Do not use an unbound writable directory or anonymous FD as the
  missing evidence channel.

## Blocking finding R3-B2 — the P2 PASS writer has no executable closure

`p2_verifier_closure_sha256` occurs exactly once as a JSON closed key (line
1455) and once as the prose phrase “P2-verifier 17-member closure hash”
(Markdown line 680). Nowhere does the amendment define its member array,
preimage tag/encoding, equality to
`candidate_conformance_index.production_executable_closure_sha256`, module
origins, runtime, command, process witness, or fresh-review binding.

The candidate snapshot rule says that the 17 production members plus the runner
are the only repository bytes executable in P2 or K7 (JSON line 1037), but that
does not identify which bytes actually implement the external verifier that
creates the controller challenge, launches the controller, constructs the
outer witness, and writes the sole PASS. Nor does it establish that the bare
field equals the 17-member digest. A closed gate-evidence object can therefore
put any 64-hex value, including 64 zeroes, in
`p2_verifier_closure_sha256`; none of the five P2 construction steps or the
gate `pass_semantics` rejects it.

This keeps the final authority transition outside the otherwise reviewed
candidate/runner/runtime graph. The later G5a statement that candidate, P2, and
S0 agree on the production closure is too late and does not make the already
issued P2 PASS well-defined.

Minimal repair: either state and enforce that
`p2_verifier_closure_sha256` equals the freshly reviewed candidate's exact
17-member production-closure digest, identify the verifier entry point and bind
its actual runtime/module/process witness, or define a separate external P2
verifier member list, byte preimage, runtime/origin closure, independent review,
and execution witness. The gate construction must recompute and compare the
value before setting `status=PASS`.

## Round 2 closure matrix

| Round 2 item | Round 3 result | Independent conclusion |
|---|---|---|
| R2-B1: seven-file closure incomplete | **CLOSED** | The member order is now the actual 17 local Python files, including both package initializers, `types`, `gates`, and `fixtures`. The exact raw-byte preimage is 316,199 bytes and hashes to `dfcf258b13aa571d3636101f09e66c9d7889a0f09dd88515f4c332ae2d662943` for the audited historical source. The 17 origin rows, package order `pams,pams.temporac`, and 34 unique sorted AST import edges agree and form an acyclic graph. |
| R2-B2: runner/observation not process-bound | **OPEN via R3-B1/R3-B2** | The runner path, one-member preimage, three roles, challenges, stdout-to-observation-root direction, exact counts, sandbox/runtime contracts, and role exchange resistance are now specified. Actual post-scoring witness transport and the PASS-writer closure remain unbound. |
| R2-B3: plan/tracker accept inner receipt | **CLOSED** | The exact plan/tracker bytes are bound by `0c77607394c413cff2d6d8909ddd1c4538462b87e037e4c70a324c398a4fc250`; every named legacy phrase is overlaid to mean the sole complete `temporac.p05m-gate-evidence.v1`. |
| R2-B4: inner receipt optional and mandatory | **CLOSED** | The six-key receipt is mandatory subordinate evidence at P2/S0/G5a/K7 and is explicitly insufficient as a standalone gate input. No optional path remains. |
| Historical/fresh separation | **CLOSED** | `historical_audit_basis` is provenance only. The future candidate index has no self hash, permits new hashes, and becomes executable only through a separate fresh review. P2 does not self-lock to historical implementation bytes. |
| K7 order | **CLOSED declaratively** | Full gate/fresh-review/snapshot/closure/runtime/witness/root validation precedes evaluator construction, process start, and any consumed-state transition. Consumption occurs only after successful OS process creation. |

## Independent fixture and science witnesses

The Markdown reference source was extracted as exactly 11,571 ASCII/LF bytes
and hashed to
`b23bf344a10f84a583c38e2326efd5ca70d4c4462fbf76715cd32b9afd20922f`.
Executing it in memory and a separately written scratch scorer produced the
same results. The local replay used Windows CPython 3.12.0 and NumPy 1.26.4,
not the future locked Linux/NumPy 2.1.0 runtime, so it is diagnostic rather than
runtime authority; every reported byte nevertheless matched the frozen
commitment.

| Witness | Independent result |
|---|---|
| fixture input canonical SHA-256 | `1264779925401d8efeecddefbe7245eca892543e923ec7c19ade08286fa062c9` |
| row count / materialized-row root | `288` / `759b344fbc7db6ea78b041c84e23ff0fc59733985775be18f6d5c235e8408dbf` |
| component-order root | `8583d714619e7b1bddc755bac95b91a801a8c97951e382fbfd7f875a70f066f2` |
| normal expected root | `50f8376d0ee1e245a8521c7831d6d23635ac8cc74694ceeef1f028c8c48e6622` |
| 17-attack expected root | `2e62a7b830f453e0505cdeabb766f8a46c0ab2523321d3b1f933bf7fd2cc5f30` |
| PCG64 seed / draws | `3783362605739069799` / `10000` |
| raw sorted-index stream | `bcfccfc96c2781162e3c973dea8f88478b1912553acc730aa52772947ed8aef1` |
| route raw-f64 stream | `8a325a51517a7a7669f4f8d4bdfa4d103204383b5d769e19cad409aba5f9e132` |
| clean raw-f64 stream | `91d1fa29d36ad2e4e41733141d841a756c986b3670617f651315026a7157bdae` |
| draw-pair root | `49b83785c436d2abd5cbcda617d142922f27ee21a4e17a39f730135add15cf3e` |
| comparator counts | global `3567`, uniform `4133`, capacity-control `2300` |
| strongest comparator | `uniform` |
| F21 route margin / clean delta | `deddddddddddbd3f` / `565555555555b5bf` |
| route percentile low/high | `2c9b6cb2c926abbf` / `e7a28b2ebae8c23f` |
| clean upper | `692fa1bd84f6a2bf` |

F20 binary64 little-endian witnesses:

| Metric | local clean/drift | global clean/drift | uniform clean/drift | capacity clean/drift |
|---|---|---|---|---|
| MAE | `efeeeeeeeeeeae3f` / `abaaaaaaaaaac63f` | `676666666666c23f` / `676666666666d43f` | `676666666666c23f` / `cdccccccccccd23f` | `555555555555c13f` / `deddddddddddd33f` |
| OBO | `888888888888e83f` / `bcbbbbbbbbbbdb3f` | `abaaaaaaaaaada3f` / `111111111111d13f` | `efeeeeeeeeeede3f` / `111111111111d13f` | `111111111111e13f` / `555555555555c53f` |

The attack order and results were exactly:

`A00 P05M_EMPTY_POPULATION`; `A01 P05M_INCOMPLETE_CROSS_PRODUCT`;
`A02 P05M_DUPLICATE_OBSERVATION`; `A03/A04 P05M_ROW_KEYS`;
`A05/A06 P05M_INTEGER_TYPE`; `A07/A08 P05M_NONFINITE`;
`A09 P05M_POSITIVE_GT`; `A10 P05M_NONNEGATIVE_ESTIMATE`;
`A11/A12 P05M_COMPONENT_ORDER`; `A13 P05M_IDENTITY_METADATA`;
`A14 P05M_SEED`; `A15 P05M_TOKEN`; `A16 P05M_MIN_COMPONENTS`.
Every attack returned only its exact `GLOBAL_FAIL` result and no partial score.

## Contract and closure hashes

| Object | SHA-256 |
|---|---|
| historical audit basis | `24a80d9007098a7a523c9634c6d5fe9f04cfb473c69b2d019fac036aca1c879c` |
| plan/tracker semantic rebinding | `0c77607394c413cff2d6d8909ddd1c4538462b87e037e4c70a324c398a4fc250` |
| production dependency DAG | `e3a6170a5d54a3bd9373939351caabeb70cbaeef5695c5aa26b3de94bb261073` |
| evaluator dependency contract | `12ae175a3934a450ddc389291d1e83152c034068de641434be0017d39ee6cc7c` |
| reference runtime contract | `3343fdcf99eba48c3c21ad2b65008855cd398cdfa04ed66d7a9ee634e9a32a3d` |
| runtime-lock contract | `d693ec66170d3e66ac1d92a049b9b57dd80ce788c38907369bbb335f426d28b1` |
| sandbox-profile contract | `a1c65c29d8482cc6f266cde3869c1f46bd57052b0d24f58972e55df0ebf08e07` |
| current historical 17-member production closure | `dfcf258b13aa571d3636101f09e66c9d7889a0f09dd88515f4c332ae2d662943` |
| reference source | `b23bf344a10f84a583c38e2326efd5ca70d4c4462fbf76715cd32b9afd20922f` |

## Complete filesystem-backed historical input manifest

The following are fresh local rehashes, not copied acceptance of author claims.

### Planning, review, and provenance inputs

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

### Current 17-member source closure

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

The proposed runner `tests/temporac/test_p05m_fixture.py` is correctly marked
not materialized; it was not treated as a current executable witness.

## Non-expansion and required disposition

The amendment preserves the F20/F21 signs and reductions, ordered three seeds,
component/video/seed order, 10,000 draws, comparator tie order and paired-clean
reuse, one claim, four experiment blocks, 27 training jobs, 93 allocated A6000
hours, thresholds, data boundary, and F23 order. It adds no natural row, model,
training job, GPU work, result, claim, or paper-visible experiment. The current
implementation remains explicitly nonconforming and no candidate or runner was
materialized by this review.

Revise the fixed/timestamped Amendment 003 pair to close R3-B1 and R3-B2, then
obtain a fresh normative review of those new bytes and, separately, the complete
materialized candidate/runtime/orchestration closure.

Same-family provisional is the maximum finding. All authority is zero:
`P0=0`, `P1=0`, `P2=0`, `P2-METRIC=0`, `P3=0`, `S0=0`, `G5a=0`, `K7=0`,
`gate=0`, `capability=0`, `evaluator=0`, `launch=0`, `server=0`, `data=0`,
`GPU=0`, `training=0`, `test=0`, `results=0`, `Git=0`, and
`paper/claim=0`. No amendment, proposal, plan, tracker, research contract,
MANIFEST, code, test, server, data, experiment, evaluator, training artifact,
or Git state was modified by this review.
