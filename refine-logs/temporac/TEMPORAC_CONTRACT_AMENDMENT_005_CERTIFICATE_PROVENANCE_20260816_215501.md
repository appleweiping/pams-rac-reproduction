# TempoRAC Contract Amendment 005 - Certificate Provenance and Closed K7 Replay (Round 7)

Status: PROPOSED_PENDING_ROUND7_REVIEW  
Family: same-family provisional  
Authority: all zero  
Generated: 2026-08-16T21:55:01+08:00

This document is the complete Round-7 author replacement for the accepted Round-6 Amendment005 specification. It is a prospective provenance contract, not an implementation, review verdict, gate receipt, test run, experiment, or authorization. It closes only the nine native Wave2 preflight findings R7-B1 through R7-B9. Every previously closed scientific, job, data, target, receipt, capability, F4, metric, threshold, and claim semantic remains unchanged.

The JSON in Appendix A is the complete machine-readable mirror and is normative with this prose. Where Round-6 native-launch prose or JSON conflicts with the exact Round-7 replacement map in Appendix A, the named Round-7 path wins. No unlisted Round-6 field is replaced.

## 1. Authority, precedence, and immutable scope

Every authority bit remains zero. This amendment grants no P0/P00, P1/P01, P2, P2-METRIC/P05M, P3/P06, S0, gate, data, server, GPU, training, test, vault, capability, evaluator, result, Git, or paper-claim authority. No supervisor, C extension, protocol entry point, wheel, environment, bundle, launch instance, discovery output, executable trace, receipt, or test described below exists merely because its schema is frozen.

Only a fresh independent Round-7 ACCEPT of the exact timestamped MD/JSON pair may permit a later, separately authorized implementation and implementation review. Until that happens, the active accepted effective-contract index is still the exact 665-byte five-row Round-6 index with SHA-256 `c203f4aab00879acf041b90829f0124d0b1de231294f191b03f3166f728e3d5c`. Current Wave1 code implements only the accepted evidence-backed pre-G5a 54-node/106-edge projection. It has no generic native Wave2 or full-DAG mode.

The scientific and resource boundary remains exact:

- one dominant TempoRAC method;
- three teacher jobs and 24 response jobs, exactly 27 training jobs;
- seeds 20260815, 20260816, and 20260817;
- 120 teacher checkpoints and 480 response checkpoints;
- 93 planned A6000-hours under the existing 100-hour ceiling;
- 268 train plus 134 development identities, exactly P402;
- four arms, three seeds, two conditions, exactly 9,648 prediction envelopes;
- the existing seven-member target NPZ and ten-key `temporac.target-receipt.v4`;
- the existing fourteen-key `temporac.g5a-receipt.v4`;
- unchanged thresholds, kill semantics, capability consumption, evaluator population, and claims.

There is no new model, loss, trainable parameter, dataset row, identity, arm, seed, condition, job, checkpoint, prediction, target member, metric, threshold, result, or claim.

## 2. Bound snapshot and byte-identical history

The accepted Round-6 fixed pair was archived byte-identically before this author revision:

| role | path suffix | bytes | SHA-256 |
|---|---|---:|---|
| Round-6 MD archive | `_20260816_215500.md` | 312,728 | `2eaa3cfb746804fdc5890fec4c252c877a031a4abedef7b9ef6eeac0a4839395` |
| Round-6 JSON archive | `_20260816_215500.json` | 261,196 | `638efc8708f74c8514247f4f4c5e2311595e90edc6744c9b371886f0789af164` |

Those bytes also equal the accepted `_20260816_184303` pair and the fixed aliases at the start of this revision. The Round-6 ACCEPT review is bound as MD `4a3199c3ac18e3fd2a5bb7ecf6d66e10bc4a5cebb82363803c47867234b70e67` and JSON `fcf8a9d4949f6b456f18bfb4591c512b6628c6ec7eb5a055873a995992f46ad6`.

Lineage keeps the immutable Round-1 `meta.json`, `request.json`, and `response.md` rows separate from `run.meta.json`: that last file is the finalized six-call Round-1-through-Round-6 review-trace aggregator, not a Round-1-only row, and is bound under its distinct aggregator role.

The final Wave1 postfix implementation review is also a non-authorizing current input: MD 13,229 bytes, SHA-256 `7cf1f1a013932a7f9317f094d28dc54af9ab0da7adb6caab4a5964941231fa3b`; JSON 12,183 bytes, SHA-256 `bdec8e1f13023e47f2be85010b7cdae09a1e219c106956a24980295bc9ce533e`. Its verdict is PASS for W1-B1/B2/B3 only and authority remains zero.

The final current Wave1 bindings independently rehashed for this author pass include:

| path | bytes | SHA-256 |
|---|---:|---|
| `src/pams/temporac/contract.py` | 22,199 | `64a579d2e31f2b46baa1442794327f19096e15ad45a1443d4b958b35e649eb99` |
| `src/pams/temporac/hashio.py` | 37,490 | `dd8c3ac3115fbd4e602e76b6106b7d0afafe1c6847fc51a6d0143b94ee73d37e` |
| `src/pams/temporac/types.py` | 46,602 | `6d575ff34ccdcadaada925e5d3b97fb59c76580cbe48ce119cec3d5ed50cf84b` |
| `src/pams/temporac/receipts.py` | 80,371 | `ab4b905d603aa7210e2953d4508d73cddefd60731d06d8315904f2a1a9267ab2` |
| `tests/temporac/test_contract_hashio.py` | 13,845 | `64c724cf1f7c622bce633eeb5c3d25c0b546afb2adedaae648bcad50f0726c29` |
| `tests/temporac/test_prediction_receipts.py` | 25,791 | `962baefc126a25203f2aec6de7257b57c1f7064cdac9283dd340046239746249` |
| `tests/temporac/test_operator_manifest.py` | 25,738 | `22072a0c6fd86889d1881b80bb41d48644c8b686c4199c934da073434b30f56a` |
| `tests/temporac/test_x0_manifest.py` | 7,139 | `d64412237c2479129b0b2bae12bd5bcf7490a0e84ab765efd798acb63dc3b9aa` |
| `pyproject.toml` | 1,750 | `14b3b35bbfa57fc982e24ef2f999df288893c0f0d12a53a9f82120babefead38` |

Appendix A binds the complete current source/test and planning/review snapshots. The canonical proposal remains 79,366 bytes, SHA-256 `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391`; the plan and tracker remain `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` and `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b`.

## 3. Canonical bytes and the non-self-referential successor contract

Canonical downstream JSON is UTF-8 without BOM, recursively sorted keys, compact comma/colon separators, `ensure_ascii=false`, `allow_nan=false`, no duplicate key, and exactly one terminal LF. Receipt and standalone-row hashes include that LF. Artifact hashes cover complete exact file bytes. Raw array hashes cover exact C-order element bytes only. Deterministic NPZ/NPY rules and tagged `H` framing remain unchanged.

The prospective Round-7 effective-contract index does not exist yet. After a fresh ACCEPT it will have the same exact `temporac.effective-contract-index.v4` two-key schema and five ordered rows:

1. exact canonical `FINAL_PROPOSAL.md`;
2. exact timestamped `_20260816_215501.md`;
3. exact timestamped `_20260816_215501.json`;
4. future exact accepted Round-7 review MD;
5. future exact accepted Round-7 review JSON.

Each row has exactly `bytes`, `role`, and `sha256`. The index has no own digest. Its external `contract_sha256` is computed only after all five immutable constituents exist. No amendment constituent contains that resulting value; no fixed alias, old review, trace, placeholder, fixpoint, or post-run backfill is a row. Existing historical science, X0, and operator artifacts retain their accepted Round-6/3.12.4 foreign keys; the successor applies only to newly built Round-7 provenance artifacts.

## 4. Preserved teacher, tune, selection, and natural-inference contract

Round 7 does not change any certificate or score arithmetic. The shared `temporac.teacher-tune-input.v4` remains the exact deterministic seven-member NPZ over the 56 fixed views `(source_id 24..31, block 0..6, linear, offset 0)`, with `N=25,680`, `E=25,624`, raw payload 22,592,544 bytes, and the existing nine-key receipt. Every one of 120 checkpoints still produces the exact two-member tune output NPZ with phase `<f4[25680,2]` and reconstruction `<f4[25680,149]`, raw payload 15,510,720 bytes, and an independent thirteen-key evaluation receipt.

The binary64 normalized SmoothL1/Huber rule remains exact. For finite binary64 `d=fl64(x-y)` and finite positive binary64 `delta`, compute `a=abs(d)`. If and only if `a <= delta`, evaluate `q=fl64(d*d)`, `h=fl64(0.5*q)`, and `loss=fl64(h/delta)`. Otherwise evaluate `h=fl64(0.5*delta)` and `loss=fl64(a-h)`. The boundary uses the quadratic branch. Every named operation rounds once under `FE_TONEAREST`; FMA, reassociation, reciprocal approximation, extended retention, vector reduction, and masked-out arithmetic are forbidden. Exact Neumaier reduction order and positive-zero canonicalization remain mandatory.

Tune abstentions remain the count of 56 full `certify_target` ABSTAIN results. Each view is one contiguous certificate run with `CertificateTrack.run_bounds=<i4[[0,N_v]]`; the ten `traversal_bounds` rows are analytic/F11 ledger partitions only and never ten certificate runs. Reconstruction F11 keeps channel -> traversal -> block -> source equal weighting. Negative fraction remains the integer ratio over traversal-owned valid-edge negative increments. The tune objective remains the already frozen exact F11 reductions and binary64 operation order.

The G1 teacher evidence still binds three teacher run receipts as exact arrays ordered by seeds 20260815, 20260816, 20260817, all 120 artifacts/receipts and all 120 tune outputs/evaluation receipts. Selection still replays the exact eight-key score rows and established per-seed-then-global precedence. A digest-only winner or missing loser is invalid.

Natural inference remains geometry-only integer landmarks -> consecutive traversal bounds -> canonical Neumaier F5 static features -> one C-order `<f4` cast -> CPU float32 `TempoRACTeacher.eval()` under inference mode, one thread, no autocast/TF32. Fewer than three landmarks immediately abstains before any teacher output or target artifact. Checkpoint loading still verifies exact deterministic uncompressed NPZ bytes, all sixteen state-dict members, strict keys/shapes/dtypes, strict fresh load, byte-identical reserialization, CPU eval/no-grad, and the exact checkpoint/selection receipts. K7 reconstructs feature, natural input, selected teacher, target NPZ, and ten-key target receipt byte-for-byte from the frozen surrounding indexes.

## 5. Preserved K1, prediction, receipt-DAG, capability, and F4 semantics

The K1 outcome index remains frozen before G5a. Its `k1_outcome_row_sha256` is SHA-256 of the exact standalone canonical thirteen-key row plus LF. Cdev is exactly the val/CERTIFIED projection of that committed index, never a later subset. The G5a certified-development index is exactly that projection, remains within the 134 development identities, and preserves K1 floors. Same-identity feature digests must equal the population-manifest row.

The population/prediction/vault identity set remains exact P402. The prediction index remains the exact `402 * 4 * 3 * 2 = 9,648` product and binds artifact, receipt, and manifest-derived component key for every identity/arm/seed/condition. NATP remains three fourteen-key receipts, each with the exact length-one K6 upstream array and complete clean/drift seed indexes and roots. K6 -> NATP -> prediction freeze -> G5a lineage remains acyclic and exact.

The accepted roster projection remains 54 owners and exactly 106 roster-direct typed edges. Its nodes include the three teacher runs, 24 response runs, X0I times three, K3, K4, K1, and NATP times three. The 57 dispatch descriptors specified later are a different ABI universe; G5A, G5B, and K7 are not silently added to the 54-node receipt roster, and 106 is not a total-edge bound on the future full G5a DAG.

Capability order is unchanged: pre-consume validates only non-vault commitments, P402/9,648/Cdev evidence; evaluator start atomically consumes; only then the same non-resumable process opens the vault, validates the exact P402 bijection/cardinality/root/period units, runs G5b, and permits K7 after G5b PASS. Any failure burns the grant. Pre-consume vault access and retry are forbidden.

F4 remains direct-composition safe. The consumer snapshots the six Mapping keys once into a plain dict, validates and canonicalizes that same snapshot, accepts a direct `MappingProxyType`, and fails a stateful mapping whose second access differs. No caller-side `dict()` adaptation is required.

## 6. R7-B1 - launch bytes and runtime-ledger bytes have exact non-circular carriers

Every embedded child uses one universal event/error pipe at FD6, one transport-only launch carrier at FD7, and one runtime-ledger bundle at FD8. Discovery has operational FDs 0,1,2,6,8; full runtime has 0 through 6 plus 8. FD7 is physical transport only: it is absent from `launch_instance.fd_rows` and from the operational sealed-memfd construction index, so launch bytes never contain the bytes/hash/evidence of the carrier that carries them.

The FD7 payload is exact canonical `temporac.native-launch-carrier.v5` with four keys: positive `launch_instance_bytes`, complete lowercase `launch_instance_bytes_hex`, `launch_instance_sha256`, and `schema`. The controller constructs it with the exact eight-stage algorithm frozen in Appendix A: create the sole writable memfd with exact `O_RDWR|O_LARGEFILE`; complete write; pread/hash/EOF verify; apply all four seals while the writer is open and verify prohibited writes; open the necessarily symlinked `/proc/self/fd/<writer>` with `O_RDONLY|O_CLOEXEC` and explicitly without `O_NOFOLLOW` as a distinct open-file description; reverify exact `O_RDONLY|O_LARGEFILE`, identity, seals, offset, and bytes; close and prove the writer dead; install the reader at FD7 with `FD_CLOEXEC=0` only for the immediately following child exec. The parent closes its copy after fork; the embedded child sets CLOEXEC before any later exec.

At native child entry, before any CPython API or project import, the child performs the exact no-new-FD `getrlimit(RLIMIT_NOFILE)` plus ascending `fcntl(F_GETFD)` scan, snapshots FD7, preads from offset zero without drifting the offset, validates count/hex/hash/canonical launch bytes, closes FD7, proves EBADF, validates FD8, and emits the exact carrier-readback evidence. Repeated finite fcntl scans prove the physical and operational FD sets without opening a self-observation `/proc/self/fd` directory. Controller construction and child readback are compared after join. Any missing seal, different OFD/inode/offset, extra FD, wrong launch hash, inclusion of FD7 in launch, or mismatched readback fails.

FD8 carries complete canonical `temporac.runtime-ledger-bundle.v5` bytes. Discovery receives exact BASE; full runtime receives exact FINAL. Controller request, launch, child pread/readback, bundle count/hex/hash, and canonical reserialization must all agree.

The effective `temporac.linux-fd-contract.v5` explicitly preserves only the accepted Round-6 constant ledger, twenty-key launch-FD row schema, null/pipe/value-domain rules, and freshness rule. It replaces the old layouts, FD3 discovery stream, close schedule, memfd applicability/name, launch, and readback schemas. Discovery operational memfds are exactly FD0 and FD8; full operational memfds are exactly FD0,3,4,5,8. Each uses the closed eight-stage v5 writer/seal/distinct-reader/install protocol and construction index, also without `O_NOFOLLOW` on the required procfs symlink. The three operational readbacks are exact post-carrier-exec-entry, locked-pre-CPython, and post-initialize indexes. Discovery closes 0/8 before preinitialization and retains 1/2/6; full closes 3/4/5/8, retains 0/1/2/6, and closes FD0 immediately after dispatch. The FD-contract and native-transport BASE artifacts are closed pointer/hash indexes over the exact immutable Amendment paths, so neither role is an informal prose digest.

## 7. R7-B2/B3 - exact failure transport, controller ownership, CLI, process order, and durability

The sole source `src/pams/temporac/_origin_supervisor.c` and sole installed ELF `/opt/temporac/replay-v4/bin/temporac-origin-supervisor` own both modes. The only controller argv is `--controller-v4`; the only embedded-child argv is `--embedded-cpython-v4`. There is no Python bootstrap, `python -I -m`, shell wrapper, daemon, RPC, editable package, or second controller image.

The controller receives one sealed FD0 request, writes only one framed FD1 result, uses an independent write-only `/dev/null` at FD2, and begins with no other descriptor. Its exact eleven-key `temporac.native-controller-request.v5` carries complete bundle bytes/count/hash, one unambiguous payload bytes/count/hash, mode, and an optional environment triple. Discovery uses BASE, null environment, and one exact `temporac.discovery-controller-input.v5` wrapper containing complete capsule and precommit bytes. Full runtime uses FINAL, a complete final environment binding FINAL, and exact `temporac.invocation.v4` bytes.

The controller initializes no CPython. It forks and absolute-path `execve`s. Between fork and exec the child uses only the frozen async-signal-safe FD actions and exec. Both discovery and full children have the same actual OS argv ending in `--embedded-cpython-v4`; only validated launch bytes select the mode. Every Round7 child deliberately selects the preserved `pyconfig_full_runtime_final_values` map because it is the only accepted 64-field map whose exact module path includes the site-packages needed by the common project/third-party import plan. The old stdlib-only discovery map remains byte-for-byte in the profile as unselected Round6 history. The v5 overlay replaces exactly four call-sequence strings at pointers 1, 2, 7, and 11 plus the native error-handling object; all 10+64 fields/defaults/setters/readbacks remain exact. Actual argv and PyConfig/sys readbacks are validated independently rather than inferred from one another. Discovery is one exact controller request, whose complete bytes/count/hash are bound once at the evidence-index top level and repeated by digest in both rows and attestations; that one request launches run 0 and then run 1 sequentially. Run 0 is drained, reaches EOF, is joined, and its complete rows remain in controller memory before run 1 exists; the combined launch/stream/join/discovery indexes are durably published only after run 1 joins, while the request, attempt marker, and each child prelaunch record follow their earlier exact durability points. Full mode launches one child. After validating and persisting a request, the controller atomically creates the exact nonretry attempt marker before any prelaunch construction or child; it remains durable on success or failure, so the same valid request cannot be retried or resume partial state. An invalid unparsed request creates no marker and grants no capability.

The FD action plan is executable rather than implicit. After persisting the request, the controller duplicates its original result pipe and stderr `/dev/null` to CLOEXEC FDs 32/33, closes 0/1/2, and opens exact low reservations at 0/1/2 so every preserved memfd writer is at least 3. It constructs every child target in the parent: sealed inputs in numeric target order, two independently opened null OFDs promoted through 40/41 and proven distinct before installation at 1/2, one pipe with controller read peer FD48 and child writer FD6, then launch and FD7. Immediately before fork only the target table plus controller-only 32/33/48 are open. The child branch closes and proves 32/33/48 dead and directly `execve`s without post-fork duplication; close/readback failure exits 125 and exec-return exits 126, which the controller maps uniquely to synthesized `FORK`/`EXEC` failure. The parent closes every target, drains 48 to EOF, joins, and only after all required children and durable evidence restores the original result/stderr OFDs from 32/33 to FDs 1/2. A failure path first closes/joins all created state and then performs the same restoration before its sole error frame; an unusable original result OFD yields no bytes and exit 124. Any scratch-fd mismatch, shared null OFD, second pipe, retained durable fd, inherited controller output/stderr, or cleanup drift fails.

The child stream is the exact v5 magic plus mode byte and length/hash-framed payloads. The exact complete success order is only the Appendix A array. Its major phases are: carrier readback; direct carrier/`PROCESS_IMAGE`/`PREINIT_NATIVE_IMAGE` normalized rows; the initial executable-mapping set; each of the three operational FD readbacks with its direct normalized row; CPython configuration plus its normalized row; the buffered initialization audit/mapping group; every precommitted import group with its optional shared mapping-delta frame; one discovery output or dispatch result; every raw/mapped shutdown row caused by `Py_FinalizeEx`; the terminal executable-mapping set; one six-key success terminal; EOF; child exit zero. Cataloged ignored-nonexecutive shutdown events remain raw evidence, while any finalization executable-origin event fails. The terminal hashes the unique earlier output and terminal mapping-set digest rather than assuming adjacency. Failure is one three-key `temporac.native-failure.v5` frame then EOF, with the exact three launch-hash phases and mutually exclusive exits: child-origin/EXEC/broken-stream failures normally exit 122, controller-origin failures exit 123, unusable FD1/FD32 exits 124, and isolated FD33 restoration is the sole specified 123 exception that may retain an earlier child stage/payload. No exception message, repr, traceback, stderr, alternate frame, or trailing byte is evidence.

Only the controller persists bytes. Its state root is the preprovisioned root-owned, mode-0700, nonsymlink `/var/lib/temporac/replay-v4/native-controller-v4`, outside the immutable `/opt` runtime. Publication uses exact `requests/attempts/prelaunch/launch/streams/joins/traces/results/attestations` directories, deterministic PID/starttime temporary names, `openat` with `O_EXCL|O_NOFOLLOW`, complete writes, `fdatasync`, `fsync`, `renameat2(RENAME_NOREPLACE)`, directory fsync, reopen, fstat, EOF, and rehash. Immutable content-addressed EEXIST is accepted only after complete equality; mutable attempt/attestation EEXIST fails. The seven-key prelaunch record binds launch, operational-memfd construction, and external FD7 construction before fork. The exact launch-evidence index has two discovery rows or one full row and adds child carrier/readback equality without output or trace. Request and attempt precede prelaunch/fork; child readback/stream/join precede final trace/result. The twelve-key durability attestation binds the attempt marker and has an exact mode-dependent present/null matrix, `state=DURABLE_REHASH_COMPLETE`, and `status=PASS`; it excludes its own digest and the later success envelope. Discovery returns the exact evidence-index bytes; full returns the exact child dispatch-result bytes. The controller emits success only after reopening and rehashing every durable object, including the full trace at its exact path.

## 8. R7-B4 - exact BASE/FINAL bundle and environment DAG

Each bundle has exactly `contract_sha256`, `rows`, and `schema`. Each row has exactly positive `bytes`, complete lowercase `bytes_hex`, unique `role`, and `sha256`. BASE has the exact nineteen-role order in Appendix A: audit catalog, CPython initialization/runtime, dispatch, effective contract, ELF, FD, floating point, initial images, installed members, kernel-runtime evidence, native build/rebuild/install/transport, import plan, wheel, source allowlist, and version layers.

Every new BASE role has a closed standalone artifact, not merely a role name. In particular, audit catalog, dispatch table, source allowlist, pre-dispatch import plan, native build recipe, FD contract, native transport, native-install index, and version-layer index each freeze exact keys, row order, canonical bytes, and an external digest. Pointer/hash indexes resolve only immutable paths in the accepted timestamped Amendment JSON and hash each complete canonical standalone value plus LF. They contain no own digest, launch evidence, final environment, or runtime-learned row.

FINAL has exactly 24 roles and begins with the byte-for-byte nineteen-row BASE prefix. It then adds capsule, two-run discovery-evidence index, common discovery output, zero-process precommit, and trusted-generated index. A bundle contains neither its own hash nor the final CPU environment. Trusted rows bind BASE only. The final v5 environment is created after FINAL and adds exact `runtime_ledger_bundle_sha256`; its future eighteen-key `runtime_launcher` instance is governed by the separate artifact schema and binds the transport digest without substituting the schema descriptor as the value. The full controller request carries environment separately. FINAL -> environment -> request -> launch is forward-only.

The exact topological order is the ordered array in Appendix A. In summary it runs accepted successor -> immutable source/project/tool/kernel/spec inputs -> reviewed build recipe and four-run preflight pipeline/plan -> authority CPython/project twin rebuild evidence -> installed/ELF/runtime/source/init/FD/FP/audit/dispatch/import/version ledgers -> BASE -> nested capsule/precommit request -> two discovery launches/carriers/streams/joins -> common output/evidence -> trusted index -> FINAL -> final environment -> the first 54 full traces -> their 54-row completion index -> G5A child-handler return and excluded durable surrounding trace -> G5B capability consumption plus same-child K7 continuation and excluded durable surrounding trace. External controller success is emitted only after child terminal/EOF, waitpid, trace/attestation persistence, reopen, and rehash. The dispatch-table artifact and `protocol_entry.py` source are sibling BASE inputs derived from the accepted specification; neither contains the other's digest.

## 9. R7-B5 - exact 57-owner byte dispatch ABI

The future Python ABI is literally:

```python
def dispatch(invocation_bytes: bytes, /) -> bytes
```

It takes exactly one positional exact `bytes` object and no keyword. The native caller imports `pams.temporac.protocol_entry`, gets `dispatch`, creates exact bytes, uses `PyObject_CallOneArg`, requires exact bytes output, copies the exact buffer, validates schema/digest, and requires byte-identical reserialization. Python exceptions are cleared and mapped only to fixed native stages; type/message/traceback never crosses the boundary.

The dispatch table has exactly 57 mechanically constructed twelve-key descriptors. The first 54 owner strings equal the accepted roster order; the final three are `G5A-COMMIT`, `G5B-JOIN`, and `K7-FINAL`. Each row freezes argument schema, callable, continuation, external-invocable bit, handler callable/token, input schema, input/result caps, owner, result schema, and side-effect profile. Handler token equals owner byte-for-byte. Default handlers are read-only content-addressed computation with no network, process, clock, random, environment, cwd, or persistence. G5B alone atomically consumes then reads the vault. K7 is not externally invocable, continues from G5B in the same process, and runs exactly once.

The historical five-key invocation keeps the field name `input_index_bytes`, but its value is complete lowercase hex. Its decoded bytes must hash and parse as the descriptor's exact index schema. Results use the exact closed inner stage result and six-key dispatch wrapper. A caller descriptor, alternate registry, owner alias, bytes subclass, oversized input/result, noncanonical wrapper, side-effect violation, external K7, or second K7 fails.

## 10. R7-B7 - pre-dispatch twin discovery covers project and third-party generation

The Round-6 three-file/dataclasses-only assumption is superseded. The bound current tree contains 89 `@dataclass` decorator sites: 87 under `pams.temporac` and two under `pams.types`. This is a source-site observation, not a predicted generated-row count.

Before either discovery child, BASE freezes the complete source allowlist, installed/wheel/CPython/ELF ledgers, audit catalog, dispatch table, exact ordered pre-dispatch import-plan artifact, and transport. The import plan loads dataclasses, NumPy, SciPy, Torch, `pams`, `pams.types`, `pams.temporac`, then the exact native-build `_fp_control` extension, all 24 Python runtime modules including future `protocol_entry`, and all transitive providers through exact installed rows. This makes the required extension-load origin and five Python readback wrappers pre-dispatch evidence in every mode. Any statically permitted failed optional import is a unique module/occurrence/reason row committed before BASE and produces raw ignored-nonexecutive evidence only; unlisted failure or discovery-learned absence fails. Discovery cannot teach or backfill BASE.

The exact six-key capsule binds BASE, contract, import plan, supervisor, and mode but no output. The exact six-key precommit embeds complete capsule bytes and `created_before_process_count=0`. It has no separate pathname: the one seven-key controller-input wrapper carries complete capsule and precommit bytes, and durable publication/reopen/rehash of the enclosing discovery request commits both nested objects before the attempt marker or any fork. It is never concatenation, two caller objects, or a later backfill.

Each of two fresh children runs that same import plan before dispatch and without data, jobs, artifacts, vault, receipts, or experiments. File-backed source compiles resolve through BASE. Each non-file-backed compile must have one observable exact source-backed Python generator frame, and its immediately adjacent exec closes one eleven-key generated row. The row records complete source bytes/hex/hash, filename, typed generator origin and its standalone hash, kind, ordinal, profile, schema, and source type. No `compile_mode` exists because stock CPython does not provide it.

Each output has exactly BASE digest, capsule digest, nonempty complete rows, and schema. Its digest and tagged root use the exact preimages in Appendix A. The evidence index has exactly two rows ordered 0,1 and binds BASE readback, launch, independent FD7 construction/readback, three operational FD readbacks, CPython configuration, complete child stream/EOF, complete output/root, process identity, wait status, and joined attestation. Outputs and rows are byte-identical; process/OFD/pipe/launch/join evidence is distinct where required.

Only after twin equality is the six-key trusted index formed. Each trusted row is the exact eleven-key output row plus `trusted_row_sha256`, where the digest is SHA-256 of the standalone eleven-key canonical row plus LF. The index binds BASE and the evidence/output roots, never FINAL. A fresh full child reruns the same import plan under FINAL and consumes each exact trusted row once as an adjacent compile/exec pair before `DISPATCH_BEGIN`; generated activity after that boundary is zero. No-frame/native, delayed, nondeterministic, missing, extra, reordered, or unmatched generation fails P06 and requires a separately reviewed predefined-method refactor; it is never guessed.

## 11. R7-B9 - stock CPython audit events map uniquely to sixteen tokens

The supervisor runtime is stock, unpatched CPython 3.12.13. Before BASE, P00 freezes a complete finite audit-policy catalog from exact CPython/native/source/install bytes and the import plan. Each catalog row fixes action, argument-projection schema, arity, emitter, event name, token-or-null, and stages. Unknown name, arity, type, stage, wildcard, runtime-learned row, executable anonymous memory, later audit-hook install, trace/profile hook, subinterpreter, Python process spawn, arbitrary marshal/pickle code, or unclassified ctypes action fails.

The exact sixteen normalized tokens remain the list in Appendix A. Six are emitted only by the native supervisor. Raw `import` has arity five and remains pending until the applicable safe barrier resolves the successful provider from `sys.modules.__spec__` and the precommitted ledger; it then becomes exactly builtin, frozen, source, or extension. Raw `compile` has arity two and no mode. Raw `exec` has one code object and is projected through a closed recursive code-object schema with no marshal, repr, pointer, or address. It becomes ordinary code exec or the adjacent trusted-generated exec. Stock exec cannot distinguish eval, so `PY_CODE_EVAL` has exact multiplicity zero. Extension loads and `ctypes.dlopen` events are buffered within the precommitted initialization or top-level `PyImport` group; only the supervisor-owned group-return barrier captures one shared before/after executable-mapping delta and attributes the exact ordered primary origins. There is no invented per-event post-return callback.

Raw argument projections are closed and typed. The exec projection binds the exact 22-key `temporac.code-object-projection.v5`; its twelve tagged constant variants freeze every key, literal tag, integer/hex/binary64/complex encoding, tuple order, frozenset ordering, recursive nested-code hash, and rejection case. Every raw row is gap-free and hashes its standalone args. Every normalized row has exactly one of thirteen v5 origin variants with an exact key set, schema token, null discriminants, event-kind map, ledger equality, and standalone canonical preimage. The no-wheel `_fp_control` load is the sole `native-build-extension` variant and can only map to `PY_EXTENSION_LOAD`. A `trusted-generated` origin resolves one exact twelve-key trusted-index row, reconstructs and rehashes its standalone eleven-key row preimage, and copies the eight explicitly mapped origin fields; `generated_source_hex` and row kind remain row-only validation fields and cannot be guessed from the trace. The normalized row hashes the six-key projection excluding its stored digest, while the separate authority projection rehashes the origin's exact provider bytes; the two digests are never conflated. Discovery candidate labels grant no authority; only twin joined evidence creates the trusted index, and only a fresh full child may consume it.

For each child the controller reserves normalized sequence zero for its own exact `PROCESS_EXEC`, but it finalizes that row only after the matching child `PROCESS_IMAGE`, terminal EOF, and zero-exit `waitpid` prove that exec; there is no implicit handshake and an exec failure has no PASS row. Because launch identity is unavailable earlier, child sequence one is the completed carrier-validated-and-closed readback; only then does the child capture `PROCESS_IMAGE`, enumerate the exact initial native map with every temporary descriptor closed, record the remaining three FD readbacks and configuration, and append safe-barrier audit mappings in actual evidence order. The same gap-free stream includes the complete raw shutdown suffix observed through `Py_FinalizeEx`; it may contain only cataloged ignored-nonexecutive events and their permitted mappings, never a new executable origin. Child stream indexes contain exact magic-through-terminal bytes and EOF evidence. The controller constructs process joins only after exact `waitpid`; a child never claims its own join and a join is not a seventeenth token. The thirteen-key final trace contains exact raw events, controller-plus-child normalized events, mapping evidence and initial/terminal mapping-set digests, process joins, launch/request/contract bindings, status, and child-stream digest, and is hashed only after assembly.

## 12. R7-B6/B8 - version split and executable native build/deploy recipe

Historical scientific/X0/operator artifacts remain CPython 3.12.4 with operator runtime lock `47c2227b74b627a03bf9b5e4930dd02484caaae5a695ae5c0129f04781115cd8`. The native provenance verifier is stock CPython 3.12.13 and may only create provenance/discovery/bytewise replay evidence. It cannot relabel or replace 3.12.4 science; any byte difference fails.

The selected CPython 3.12.13 source archive is 20,801,708 bytes with SHA-256 `c08bc65a81971c1dd5783182826503369466c7e67374d1646519adf05207b684`. The selected project builder is Hatchling 1.27.0 wheel SHA-256 `d3a2f3567c4f926ea39849cdf924c7e99e6686c9c8e288ae1037c8fa2a5d937b`. Its exact build environment uses `SOURCE_DATE_EPOCH=315532800`, the UTC epoch for 1980-01-01, so every ZIP timestamp is representable by this fixed Hatchling/ZIP implementation; epoch zero is forbidden. The selected native toolchain is GCC 13.2.0, GNU ld 2.41, target `x86_64-linux-gnu`, ISA `x86-64-v2`, at the exact absolute paths in Appendix A. The complete tool ledger must populate non-null exact binaries, headers, crt objects, libraries, image, versions, byte counts, and hashes before P00/P06; this author document fabricates none.

Appendix A freezes every configure, compile, link, wheel, extraction, and twin-build argv token; cwd, environment, umask, network prohibition; C17 and floating-point flags; PIC/PIE, RELRO, NOW, noexecstack, build-id, RPATH/DT_NEEDED, unstripped/no-text-relocation acceptance; and deterministic twin equality. The exact six-key native-build-recipe artifact carries five reviewed digests: accepted contract, closed immutable spec-pointer index, source allowlist, the separate project-build-input index containing the exact root `README.md` bytes required by `pyproject.toml`, and the complete future toolchain ledger; its sixth key is the schema. Four non-authoritative clean-image preflights run first: two CPython preflights retain and compare complete stage/output/exec/input/temp/stream evidence; only their validated run-0 stage is mounted read-only into two project-native preflights; the four run-independent projections then form the independently accepted two-kind exec plan. Authority-bearing CPython twins and, only after their equality, project-native twins must match that plan and their own complete per-run evidence. These provenance build validations are not experiment jobs or GPU hours. No PATH compiler/linker or unledgered PATH image, shell expansion in native compilation, `python-config`, `pkg-config`, LTO, PGO, fast-math, native march, stripping, editable install, or post-link mutation is allowed.

Both CPython builds run in two nonoverlapping writable instances of the same immutable image but see byte-identical internal absolute source/build/stage paths. This prevents stock configure's installed Makefile from embedding `-0` versus `-1` build roots. Their complete exported member ledgers must be byte-identical. Deterministic checked-hash stdlib pyc created during staging is allowed only in the exact two-build-equal exclusion index and is not deployed; runtime write-bytecode is zero, so marshal-loaded runtime code remains absent. Each later project-native rebuild instance materializes the validated authority run-0 CPython export at the same read-only `/build/temporac/cpython-stage`; CPython headers/libpython resolve only there, while every non-CPython system/compiler header, crt object, and library resolves only through the exact toolchain/build-image ledger, and final ELF RPATH names `/opt/temporac/replay-v4`. Exactly out/tmp/cache are writable for project-native builds; home is mode 0500 and nonwritable. The exact command order builds the wheel, supervisor object, hidden native FP object, supervisor PIE, extension object, and extension ELF; the native FP intermediate is linked into the supervisor and never installed separately. Deployment copies the exact CPython complement, deterministically maps the project wheel and every exact accepted third-party wheel member into its predicted purelib/platlib/scripts/headers/data path without installer execution or rewrites, then overlays only the supervisor and `_fp_control` outputs. It requires complete native-install equality, fsyncs, atomically renames, and reopens/rehashes the full installed ledger.

`protocol_entry.py` is a future noneditable project-wheel member equal to its repository allowlist row. `_fp_control.cpython-312-x86_64-linux-gnu.so` is installed at the exact `site-packages/pams/temporac` path as `native-build-output/no-wheel-member/no-repository-member`; the supervisor is likewise a native build output. The C source purposes are closed: `_origin_supervisor.c` owns transport/controller/audit/CPython embedding and calls, but does not duplicate, the linked native FP ABI; `_fp_control.c` compiles under two mutually exclusive macros into the supervisor's hidden five-function native ABI and the extension's five Python wrappers from the same source-level operations. Before any CPython API the supervisor applies and reads back FE/MXCSR through the linked ABI; later extension calls and Torch probes verify the same preserved control semantics. Neither source changes teacher, target, certificate, metric, or evaluator computation. These three future sources are presently absent, so execution is blocked until a separately authorized implementation and review; their absence is not hidden by a current-tree fallback.

## 13. Required future positive and negative tests

No test was added or run in this author revision. After ACCEPT and separate implementation authority, tests must cover at least:

- missing/tampered/unsealed/wrong-OFD FD7, inclusion of FD7 in launch, wrong FD8 bundle, offset drift, operational memfd/readback/close-stage mismatch, forbidden procfs `O_NOFOLLOW`, extra FD, and exact successful carrier plus operational eight-stage/readback equality;
- pre-carrier null versus post-carrier exact launch digest, every truncated/reordered/duplicate/trailing frame, EOF/exit mismatch, broken stream, exec/signal synthesis, and exact success terminal;
- wrong controller/child image, argv, cwd, env, uid/gid, umask, FD action, old stdlib-only discovery PyConfig selection, profile-overlay pointer/value drift, parallel/reused twin, absent/reused attempt marker, retry, nondurable prelaunch/trace publication, EEXIST mismatch, self-referential attestation, and clean durable success;
- missing/extra/reordered BASE or FINAL role, FINAL-prefix drift, environment in bundle, trusted row binding FINAL, bundle/environment/launch/readback mismatch, and a positive acyclic DAG simulation;
- every bytes-ABI type/arity/keyword/owner/handler/input/result/cap/exception/side-effect failure, direct valid dispatch, G5B capability consumption, external or duplicate K7 rejection, and exact same-process G5B -> K7 success;
- 3.12.13 science substitution, 3.12.4 supervisor substitution, historical foreign-key rewrite, and the valid evidence-only version split;
- missing/tampered/reordered project or third-party generated row, empty/twin-different output, no-frame/native/delayed generation, post-dispatch generation, discovery feedback into BASE, and exact two-fresh-child/full-consumption success;
- toolchain/image/header/library/hash/flag mismatch, missing/extra/reordered build-exec row, unaccepted preflight, fast-math/LTO/strip/shell discovery, missing/separately installed native FP object, FP compile-mode/ABI/readback drift, unequal rebuild, staged-sysroot drift, bad pyc exclusion, editable/wheel mismatch, false `_fp_control` wheel mapping, ELF dynamic-tag mismatch, and exact deterministic build/deploy ledgers;
- unknown raw audit name/arity/type, premature provider classification, eval guess, repr/pointer/mode injection, unsafe code/marshal/ctypes event, event omission/duplication, omitted/reordered finalization shutdown event, post-output executable origin, child-recorded join, token 17, and exact safe-barrier/final-trace reconstruction;
- all previously required score swap, missing loser, checkpoint member, code/environment, K1 omission, feature swap, component relabel, vault ordering/retry, receipt-root DAG self/extra/missing, negative zero, owner-key typed-preimage, direct MappingProxy, target byte equality, and cardinality tests.

## 14. Round-7 closure matrix

| finding | closure | exact result |
|---|---|---|
| R7-B1 | EXACTLY_SPECIFIED | FD7 carries complete launch bytes/hash before CPython, has independent eight-stage evidence, and is excluded from launch; operational memfd/readback transitions and FD8 BASE/FINAL bytes are closed. |
| R7-B2 | EXACTLY_SPECIFIED | Universal FD6 plus exact success/failure framing, three-key error record, EOF, exit, and controller synthesis closes every observable failure boundary. |
| R7-B3 | EXACTLY_SPECIFIED | One source/ELF owns exact controller/child modes, request, argv, environment, FD, process order, nonretry attempt marker, prelaunch/trace paths, and durable no-overwrite publication. |
| R7-B4 | EXACTLY_SPECIFIED | Closed byte-carrying 19-row BASE and 24-row FINAL, separate final environment, exact carriers, and topological exclusions remove path-only and self-cycle ambiguity. |
| R7-B5 | EXACTLY_SPECIFIED | Literal exact-bytes ABI, mechanical 57-row descriptors, result wrappers, C call sequence, exception mapping, and side-effect profiles close dispatch. |
| R7-B6 | EXACTLY_SPECIFIED | Historical 3.12.4 science and stock 3.12.13 provenance verification are distinct typed layers; no silent replacement exists. |
| R7-B7 | EXACTLY_SPECIFIED_FAIL_CLOSED | BASE-precommitted project/third-party import plan, twin nonempty discovery, exact evidence/trusted schemas, and fresh full consumption cover observable generation; unobservable generation fails. |
| R7-B8 | EXACTLY_SPECIFIED_FUTURE_VALUES_REQUIRED | Toolchain, complete preaccepted build-exec index, commands, staged sysroot, flags, two-build equality, dual-mode FP native/extension ABI, and install/ELF/wheel/native associations are frozen; actual future bytes/hashes must be populated and reviewed, never invented. |
| R7-B9 | EXACTLY_SPECIFIED | Closed raw catalog/projections, stock arities, safe barriers, zero eval mapping, sixteen tokens, thirteen typed origins, evidence-finalized controller exec rows, complete finalization suffix, child streams, and controller joins uniquely close the trace. |

## 15. Final disposition

The disposition is `PROPOSED_PENDING_ROUND7_REVIEW`, same-family provisional, authority all zero. This author revision does not claim current native implementation, runtime artifacts, evidence, tests, or full-DAG support. It changes no source, tests, proposal, plan, tracker, review, MANIFEST, trace, server, data, sealed/heldout/vault/result state, or Git metadata.

No fresh review, implementation, test, experiment, server action, capability action, or launch was started. A future reviewer must validate the exact timestamped pair, all R7-B1 through R7-B9 closures, the preserved 10+64 CPython initialization and floating-point profiles, the 54-owner/106-edge evidence projection, all scientific locks, and the no-self-cycle successor construction before any implementation authority can be considered.

## Appendix A - complete machine-readable mirror

```json
{
  "amendment_id": "TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE",
  "amendment_round": 7,
  "authority": {
    "P2": 0,
    "P3": 0,
    "S0": 0,
    "capability": 0,
    "data": 0,
    "evaluator": 0,
    "gate": 0,
    "git": 0,
    "gpu": 0,
    "heldout": 0,
    "launch": 0,
    "p0": 0,
    "p1": 0,
    "p2_metric": 0,
    "paper_claim": 0,
    "results": 0,
    "sealed": 0,
    "server": 0,
    "test": 0,
    "training": 0
  },
  "bindings": {
    "accepted_wave1_postfix_non_authorizing": {
      "active_effective_contract_bytes": 665,
      "active_effective_contract_sha256": "c203f4aab00879acf041b90829f0124d0b1de231294f191b03f3166f728e3d5c",
      "authority": 0,
      "implementation_files": {
        "src/pams/temporac/contract.py": "64a579d2e31f2b46baa1442794327f19096e15ad45a1443d4b958b35e649eb99",
        "src/pams/temporac/hashio.py": "dd8c3ac3115fbd4e602e76b6106b7d0afafe1c6847fc51a6d0143b94ee73d37e",
        "src/pams/temporac/receipts.py": "ab4b905d603aa7210e2953d4508d73cddefd60731d06d8315904f2a1a9267ab2",
        "src/pams/temporac/types.py": "6d575ff34ccdcadaada925e5d3b97fb59c76580cbe48ce119cec3d5ed50cf84b"
      },
      "review": {
        "json_bytes": 12183,
        "json_sha256": "bdec8e1f13023e47f2be85010b7cdae09a1e219c106956a24980295bc9ce533e",
        "markdown_bytes": 13229,
        "markdown_sha256": "7cf1f1a013932a7f9317f094d28dc54af9ab0da7adb6caab4a5964941231fa3b",
        "verdict": "PASS"
      },
      "scope": "W1-B1/B2/B3 only; exact evidence-backed pre-G5a 54-node/106-edge projection; no future full DAG or native Wave2 implementation"
    },
    "current_runtime_bootstrap_snapshot_non_authorizing": {
      "README.md": "cbd538dddb0f4b70774e674710f279877ecfae37abeff58a0b32c555583a8ee3",
      "pyproject.toml": "14b3b35bbfa57fc982e24ef2f999df288893c0f0d12a53a9f82120babefead38",
      "src/pams/__init__.py": "8b6ffbf3ac8ed7e43f1f40ef5793baa087f73b303eff342d891380bc9a5d67ce",
      "src/pams/types.py": "78c2ad53e98856425492a1711faa62fc2bb7764767dc48a22f7c4d5e043a3e31"
    },
    "current_source_snapshot_non_authorizing": {
      "src/pams/temporac/__init__.py": "e3f71c2d85974ac3f635f76ee61f8e1f793b9bc251b25ff4336563d4da52c2a3",
      "src/pams/temporac/certify.py": "468ce8418bff0f43582d341f3e1b8064cf1c429301d1f56be84b8e13ad59a3f1",
      "src/pams/temporac/contract.py": "64a579d2e31f2b46baa1442794327f19096e15ad45a1443d4b958b35e649eb99",
      "src/pams/temporac/cue.py": "702784bef27db94c1dc027d036a64d7461ae618613baec5e68061254f98569c7",
      "src/pams/temporac/decode.py": "766cd6ca4b9e133041bd602f50fde840ec7caadd376bd6851799583d44f31f5e",
      "src/pams/temporac/evaluator.py": "f981c6c7061c3388f06ef39466ef98192e65c62c501821be2b211f903cfaf9de",
      "src/pams/temporac/feature_io.py": "f54f17ca204c242d304314f4280e7540c28588e86ea588a405fff02176b95428",
      "src/pams/temporac/fixtures.py": "979b8cee4fed117f5471cf399794f615c60eae52dbba2c946109ff836206b464",
      "src/pams/temporac/gates.py": "b6c9d7432443d7ad36738a81700746709fd571d475514aa6f1bbcd39be744b4a",
      "src/pams/temporac/hashio.py": "dd8c3ac3115fbd4e602e76b6106b7d0afafe1c6847fc51a6d0143b94ee73d37e",
      "src/pams/temporac/metrics.py": "1dadf08e89330bed2ab773afda05bffcb3fe2252b0d3e23c0bcc18c2c3b049dd",
      "src/pams/temporac/nola.py": "ad6a13203f09c13a37a854333d4458ca8ce5a03bb186c0946f014a4e8cb9a637",
      "src/pams/temporac/objective.py": "09a77231a48a869cbc83e0aa1aa2502f86f4abbcbab8c84f2b2c5abbd729ac87",
      "src/pams/temporac/prediction.py": "7c57725240e83b7c06227a3b27e5f3cadb4590170488e5e126180caba8c13808",
      "src/pams/temporac/preprocess.py": "042b5cae1917359cb5596837c81c0e36658ff99a6a8fd6f2602b478ca6f98bc3",
      "src/pams/temporac/quadrature.py": "44f4b91dc4d5631442982732177a6f57726c0b8f42594380b5a75f2209efc2f3",
      "src/pams/temporac/receipts.py": "ab4b905d603aa7210e2953d4508d73cddefd60731d06d8315904f2a1a9267ab2",
      "src/pams/temporac/response.py": "04df32d6e2e4061d86d813f611fb9ff277f2be8312af4efbc7219785af02f9bd",
      "src/pams/temporac/runtime.py": "9421661046ee686508607529ed972b087e724c2c3a5c487e220c2f8e4038a0c8",
      "src/pams/temporac/teacher.py": "30575fa64875aa6b77de157cf6af348b3b7900851a1c1742993842eea7928b30",
      "src/pams/temporac/training.py": "0af8ecbadfcda43d2b9f60853c99210b316786ff4310ed5d0a32d9e366f627a4",
      "src/pams/temporac/trusted_packer.py": "40d6e3d5be1b26b7ba0f0f7aadf8da7496c95c4e829face5c7e9faaee2cdf94e",
      "src/pams/temporac/types.py": "6d575ff34ccdcadaada925e5d3b97fb59c76580cbe48ce119cec3d5ed50cf84b",
      "src/pams/temporac/x0.py": "5f4585425a83bf63779265db9004d210c5c2875a579ebc9a6c51297eb647c7a2"
    },
    "current_test_snapshot_non_authorizing": {
      "tests/temporac/__init__.py": "dd07192e3b73d582dfb5f0325cfafd7462040c76574ad8241a661fdc4713a24e",
      "tests/temporac/test_certificate_artifact.py": "f2b0ef6e6d977bf3532595147fba89b611258063ee5e36e30657c31c62e56c20",
      "tests/temporac/test_certificate_tau.py": "f7e2f25ab93226568910b856167e441037213ba3dd937d6915587037b7c6a94b",
      "tests/temporac/test_contract_hashio.py": "64c724cf1f7c622bce633eeb5c3d25c0b546afb2adedaae648bcad50f0726c29",
      "tests/temporac/test_feature_io_firewall.py": "f049e54c7254a848dc6cdd10afb853c5538fd8ecabd02b86950eb77d8a74c12b",
      "tests/temporac/test_gates_fixtures.py": "d6d91a7dd606c42f672bb015d77c5c10406c2c6de69ed3a8bff76a0ce4f713d3",
      "tests/temporac/test_metrics_evaluator.py": "a6f4aa275796b06162cac863674205d85b07adc46f89bc8d9821c48d17440fc6",
      "tests/temporac/test_nola_decode.py": "80586866a9293e892c2cab1ca7318b97d341c2aa5990f7e1a31695f18376ff0b",
      "tests/temporac/test_objective_runtime.py": "cc3685e2bf53ed7535addb1c09f08e0520dbd9143ba5d44da1cca28761107fa5",
      "tests/temporac/test_operator_manifest.py": "22072a0c6fd86889d1881b80bb41d48644c8b686c4199c934da073434b30f56a",
      "tests/temporac/test_prediction_identity.py": "7967b11ee6585c2664fa4f39f5a9d9798ab407317f0d4efcf790f8d7a7bd5148",
      "tests/temporac/test_prediction_receipts.py": "962baefc126a25203f2aec6de7257b57c1f7064cdac9283dd340046239746249",
      "tests/temporac/test_preprocess_geometry.py": "2edda5c593992c32aca30452340fce357930262978e331d37b6e5d55d3eb3b3f",
      "tests/temporac/test_quadrature_cue.py": "4b7f977cbd31dc441ac021666bbc036b1d7de505ee884445df356b57b88374b6",
      "tests/temporac/test_response_invariants.py": "b2e88e011bb06b9e21011ff252b630ffcbfcbe53a29638f8983b563f00ffae63",
      "tests/temporac/test_teacher_certificate.py": "a5dc0bff236f417ab570193e46542b693d0f201fedd91693e6c572a70d20e012",
      "tests/temporac/test_training_fresh_graph.py": "52b9f266173fec04152240d629dbf651aadd5b85d5839abcfe9f16fedc608845",
      "tests/temporac/test_x0.py": "90206f53c7fffeabe9bcb3d6374fd60d3c51c69a7d1137df0cba797b583d1ef8",
      "tests/temporac/test_x0_manifest.py": "d64412237c2479129b0b2bae12bd5bcf7490a0e84ab765efd798acb63dc3b9aa"
    },
    "planning_and_review": {
      "refine-logs/EXPERIMENT_PLAN.md": "4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e",
      "refine-logs/EXPERIMENT_TRACKER.md": "714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b",
      "refine-logs/temporac/FINAL_PROPOSAL.md": "3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391",
      "refine-logs/temporac/TEMPORAC_CERTIFICATE_IMPLEMENTATION_REVIEW_POSTFIX_ROUND2_20260816.json": "144d2f22f4f75166b8cbefe54157d08690408979dd7cb34ef8c97e1738432872",
      "refine-logs/temporac/TEMPORAC_CERTIFICATE_IMPLEMENTATION_REVIEW_POSTFIX_ROUND2_20260816.md": "93a9cae771b34c185b9057286794b894d615aec5424011df145aaf8900f008d8",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND2.json": "ed877ba3d8fd15c76c9908dd694bfbb804ed0c7b05f99a97853dc55c51ea56bd",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND2.md": "9d401fcde35786bfe2925cf31a803bd515668a156f08ae5bc9ef5c27810a4101",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND3.json": "df7917f5eb215779afca36a4db1b978f0a2f6b64b720267b2b14cb20e703339a",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND3.md": "e29f1cab1e92feba575cf7d5f446de4f8388b42b7c2d1c65e21d8187f856a54e",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND4.json": "f8552023abb52c98baec6a9143d15e93e3268d883bf3967c446c3fcbcd9d06d9",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND4.md": "988eb7c2b61eb445a07e1c17b7bcf44be256f9948d6600a6ed6edc62ef71cfc1",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND5.json": "8f2490fcfe8434e1d3f2adc3a2a5db7afb4166302f421fa06e45512b764cf2ff",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND5.md": "ca3807657ee4197996c49c270cabd73bb43e439422173ca780849f433113a6f7",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND6.json": "fcf8a9d4949f6b456f18bfb4591c512b6628c6ec7eb5a055873a995992f46ad6",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND6.md": "4a3199c3ac18e3fd2a5bb7ecf6d66e10bc4a5cebb82363803c47867234b70e67",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH.json": "e443453b93c96dfda4c6daf68332c0f667429cbf2a036e16e0e6c43c89d9fa61",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH.md": "9b1e3a9c36c360e8d9011145d120f7cff8e6922f4620c60445d8194809070517",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH_REVIEW_ACCEPTED.json": "b59d381b5e515b7657861f636d2a50647faae7b18b2c73341550547109b57353",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH_REVIEW_ACCEPTED.md": "ef84a0a3fde4a0269b34ba37fb2a4d41f7164a5382f8f3efaf4dc5948716388f",
      "refine-logs/temporac/TEMPORAC_WAVE1_CONTRACT_IMPLEMENTATION_POSTFIX_REVIEW_20260816.json": "bdec8e1f13023e47f2be85010b7cdae09a1e219c106956a24980295bc9ce533e",
      "refine-logs/temporac/TEMPORAC_WAVE1_CONTRACT_IMPLEMENTATION_POSTFIX_REVIEW_20260816.md": "7cf1f1a013932a7f9317f094d28dc54af9ab0da7adb6caab4a5964941231fa3b",
      "refine-logs/temporac/TEMPORAC_WAVE1_CONTRACT_IMPLEMENTATION_REVIEW_20260816.json": "04343f3f93ad0cb91a70621a26032ec16414182a5b1262aa9b8e19535cbd0698",
      "refine-logs/temporac/TEMPORAC_WAVE1_CONTRACT_IMPLEMENTATION_REVIEW_20260816.md": "0b160e9f88264769ea91e586708173c06468e56f22d7533121c233d830b05ef9"
    }
  },
  "canonical_bytes": {
    "H": "SHA256(tag_ASCII || 0x00 || for each field: uint64_be(byte_length) || field_bytes)",
    "array_raw_hash": "SHA-256 over exact C-order element bytes with no filename, shape prefix, NPY header, or LF",
    "artifact_hash": "SHA-256 over the complete exact artifact bytes",
    "float_rules": "finite IEEE-754 binary64 only in JSON score fields; parsed binary64 bits must equal recomputation; mathematical zero is encoded as +0.0 and -0.0 is rejected",
    "hash_format": "64 lowercase hexadecimal SHA-256",
    "integer_rules": "range-check before encoding; counts and byte lengths are JSON integers; no float substitute",
    "json": "UTF-8 without BOM; object keys lexicographically sorted; separators comma and colon; ensure_ascii=false; allow_nan=false; duplicate keys forbidden; exactly one terminal LF",
    "member_hash": "SHA-256 over the complete exact NPY-v2.0 member bytes",
    "member_ledger_closed_keys": [
      "bytes",
      "dtype",
      "name",
      "sha256",
      "shape"
    ],
    "npz": "unique deterministic ZIP_STORED archive; flat ASCII member names with .npy suffix in bytewise ASCII order; NPY v2.0; allow_pickle=false; fixed DOS 1980-01-01 timestamp; flags, extra, comment, external attributes all zero; no duplicate, directory, encrypted, compressed, ZIP64, trailing, or extra member; load then reserialize must be byte-identical",
    "receipt_hash": "SHA-256 over the complete canonical JSON bytes including the single terminal LF"
  },
  "closed_indexes_and_roots": {
    "aggregate_preimage_rule": "Every H field is the complete exact canonical index bytes unless explicitly stated raw32 or ASCII. A digest list, filesystem enumeration, path string, object iteration order, or caller count is never a substitute.",
    "certified_development_index": {
      "equality": "The certified-index identity order, target-count identity order, and evaluator actual typed CertifiedDevelopmentTarget evidence order all equal exact Cdev. Row identities, component keys, feature receipts/artifacts, natural inputs, selection, selected teacher, and target bytes equal corresponding K1/population/target evidence. No caller subset and no post-K1 filtering.",
      "identity_set": "exact Cdev = exact val/CERTIFIED projection of the frozen K1 outcome index",
      "order": "opaque key bytes then slot",
      "row_closed_keys": [
        "component_key_hex",
        "feature_artifact_sha256",
        "feature_receipt_sha256",
        "k1_outcome_row_sha256",
        "natural_input_receipt_sha256",
        "opaque_key_hex",
        "population_row_sha256",
        "selection_receipt_sha256",
        "selected_checkpoint_artifact_sha256",
        "selected_checkpoint_receipt_sha256",
        "slot",
        "target_artifact_sha256",
        "target_receipt_sha256"
      ],
      "schema": "temporac.certified-development-index.v4",
      "target_count": "Existing target_count_root rows are exactly this identity order projected to opaque_key_hex, slot, and integer sum of pulse bits from the bytewise-verified target.",
      "top_level_closed_keys": [
        "contract_sha256",
        "k1_outcome_index_sha256",
        "population_manifest_sha256",
        "rows",
        "schema"
      ]
    },
    "checkpoint_index": {
      "exact_inventory": "120 teacher rows at steps 500..20000 plus 480 response rows at steps 500..10000",
      "family_rules": "family is exact ASCII teacher or response; teacher rows have exact checkpoint/evaluation digests and agree with the 120-row evidence index; response rows use null for those three surrounding fields and are bound by their exact existing run receipt ordered_checkpoint_sha256 entry",
      "order": "decoded ASCII job_name_hex ascending, then numeric step ascending; all 600 rows exactly once",
      "replay": "At G1 and K7 all 120 teacher artifacts are supplied and every one of 16 members is parsed/rehashed/shape-dtype checked/strict-loaded/reserialized; every tune output/evaluation is replayed. No aggregate digest or winner-only path replaces any exact artifact. All 480 response rows are checked against the exact 24 completed run-receipt checkpoint arrays.",
      "root": "checkpoint_root_sha256=H(temporac.checkpoint-root.v4, exact canonical checkpoint-index bytes including LF); tag is exact ASCII and H uses the canonical uint64 length framing",
      "row_closed_keys": [
        "artifact_sha256",
        "checkpoint_receipt_sha256_or_null",
        "family",
        "job_name_hex",
        "run_receipt_sha256",
        "seed",
        "step",
        "tune_evaluation_artifact_sha256_or_null",
        "tune_evaluation_receipt_sha256_or_null"
      ],
      "row_count": 600,
      "schema": "temporac.checkpoint-index.v4",
      "top_level_closed_keys": [
        "contract_sha256",
        "rows",
        "schema",
        "score_index_sha256",
        "selection_receipt_sha256",
        "teacher_checkpoint_evidence_index_sha256"
      ]
    },
    "feature_index": {
      "order": "population order",
      "root": "feature_root_sha256=H(temporac.feature-root.v4, exact feature-index bytes)",
      "row_closed_keys": [
        "feature_artifact_sha256_or_null",
        "feature_receipt_sha256_or_null",
        "opaque_key_hex",
        "population_row_sha256",
        "slot",
        "split"
      ],
      "row_count": 402,
      "schema": "temporac.feature-index.v4",
      "semantics": "Each nonnull receipt digest equals the same population row and yields the named artifact digest; null pairs match exactly. No extra feature receipt is admitted.",
      "top_level_closed_keys": [
        "contract_sha256",
        "population_manifest_sha256",
        "rows",
        "schema"
      ]
    },
    "g5a_root_index": {
      "index_classes_ascii_order": [
        "certified-development",
        "checkpoint",
        "executable-origin-trace",
        "feature",
        "k1-outcome",
        "natural-prediction-completion",
        "pre-g5a-receipt-inventory",
        "prediction-clean",
        "prediction-drift",
        "score",
        "target",
        "teacher-checkpoint-evidence"
      ],
      "index_count": 12,
      "index_row_closed_keys": [
        "bytes",
        "class",
        "sha256"
      ],
      "receipt_root": "existing G5a receipt_root_sha256=H(temporac.receipt-root.v4, exact complete g5a root-index bytes including LF)",
      "schema": "temporac.g5a-receipt-root-index.v4",
      "top_level_closed_keys": [
        "contract_sha256",
        "indexes",
        "receipt_dag_bytes",
        "receipt_dag_sha256",
        "schema"
      ],
      "validation": "For every row the verifier receives exact canonical index bytes, requires positive byte count and digest equality, exact one-row-per-listed-class multiplicity, ASCII class order, schema, cross-index foreign keys, cardinality, and no unlisted index. The executable-origin row is the exact pre-G5a completion index; the NATP row binds exact pilot scope, K6, three NATP receipts, six seed indexes/roots, and both final prediction roots; the receipt-inventory row is the exact 54-owner index."
    },
    "natural_prediction_completion_index": {
      "byte_fields": "Every *_bytes field is the exact positive canonical byte count of the separately supplied artifact; every paired SHA is over those complete bytes including LF.",
      "lineage": "Every row repeats the one exact K6 receipt and pilot-scope digest, binds one NATP receipt, and binds both per-seed condition indexes/roots. The completion index bytes/hash plus typed NATP receipts/DAG edges are committed by G5a before capability consumption.",
      "merge_rule": [
        "For each condition, validate the three seed indexes and roots from the three rows.",
        "The existing temporac.prediction-index.v4 is the exact deterministic merge in population order, arm order, then seed order 20260815/16/17. Every merged row is byte-for-byte equal to its one source seed row; no recomputation, omission, duplicate, or resort.",
        "Recompute the existing condition root from the exact merged bytes. The top-level clean/drift index and root digests must equal those recomputed values."
      ],
      "row_closed_keys": [
        "clean_seed_index_bytes",
        "clean_seed_index_sha256",
        "clean_seed_root_sha256",
        "drift_seed_index_bytes",
        "drift_seed_index_sha256",
        "drift_seed_root_sha256",
        "k6_receipt_sha256",
        "natp_receipt_sha256",
        "pilot_scope_manifest_sha256",
        "seed"
      ],
      "row_count": 3,
      "row_order": "seed numeric ascending: 20260815, 20260816, 20260817",
      "schema": "temporac.natural-prediction-completion-index.v4",
      "seed_index": {
        "exact_product": "402 identities x four arms for exactly one seed and one condition",
        "order": "population manifest order, then arms local/global/uniform/capacity-control; seed and condition fixed by the index",
        "row_closed_keys": [
          "arm",
          "artifact_sha256",
          "component_key_hex",
          "condition",
          "opaque_key_hex",
          "population_row_sha256",
          "receipt_sha256",
          "seed",
          "slot",
          "split"
        ],
        "row_count": 1608,
        "schema": "temporac.prediction-seed-index.v4",
        "top_level_closed_keys": [
          "condition",
          "contract_sha256",
          "population_manifest_sha256",
          "rows",
          "schema",
          "seed"
        ]
      },
      "seed_root": {
        "preimage": "H(tag, condition ASCII, uint64_be(seed), exact canonical seed-index bytes including LF)",
        "tag": "temporac.prediction-seed-root.v4"
      },
      "top_level_closed_keys": [
        "contract_sha256",
        "k6_receipt_sha256",
        "pilot_scope_manifest_bytes",
        "pilot_scope_manifest_sha256",
        "prediction_clean_index_sha256",
        "prediction_clean_root_sha256",
        "prediction_drift_index_sha256",
        "prediction_drift_root_sha256",
        "rows",
        "schema"
      ]
    },
    "pilot_scope_manifest": {
      "closed_keys": [
        "arms",
        "conditions",
        "contract_sha256",
        "development_identity_count",
        "development_video_count",
        "labels",
        "population_manifest_sha256",
        "protocol",
        "seeds",
        "schema",
        "train_identity_count",
        "train_video_count"
      ],
      "fixed_values": {
        "arms": [
          "local",
          "global",
          "uniform",
          "capacity-control"
        ],
        "conditions": [
          "natural-clean",
          "natural-drift"
        ],
        "development_identity_count": 134,
        "development_video_count": 51,
        "labels": [
          "partial-cache",
          "GT-bbox-assisted",
          "supplied-track"
        ],
        "protocol": "GT-bbox-assisted AlphaPose supplied-track pilot",
        "seeds": [
          20260815,
          20260816,
          20260817
        ],
        "train_identity_count": 268,
        "train_video_count": 110
      },
      "rules": "The exact canonical manifest has these values and the exact population_manifest_sha256 for P402. It contains no evaluator-vault count/period/density fields, no human metric, and no main-paper eligibility; it is appendix-only surrounding scope.",
      "schema": "temporac.pilot-scope-manifest.v4"
    },
    "population": {
      "artifact": "existing exact 402-row population manifest",
      "cardinality": "268 train plus 134 val equals 402",
      "digest": "population_manifest_sha256=SHA-256(exact canonical manifest bytes)",
      "order": "train then val, opaque key bytes, slot",
      "row_keys": [
        "component_key_hex",
        "eligible",
        "feature_receipt_sha256_or_null",
        "opaque_key_hex",
        "reason_codes",
        "slot",
        "source_binding_sha256",
        "split"
      ]
    },
    "prediction_indexes": {
      "arms": [
        "local",
        "global",
        "uniform",
        "capacity-control"
      ],
      "combined_row_count": 9648,
      "component_rule": "component_key_hex and population_row_sha256 are derived inside the index builder and evaluator from the exact manifest row. No prediction envelope/API caller component token is accepted; all 24 envelopes for an identity inherit one identical manifest component.",
      "conditions": [
        "natural-clean",
        "natural-drift"
      ],
      "exact_product": "402 identities x 4 arms x 3 seeds x 2 conditions, one artifact and receipt per cell, no missing/extra/duplicate",
      "order": "population order, then arm in listed order, seed ascending; condition is fixed by its index",
      "per_condition_row_count": 4824,
      "roots": [
        "prediction_clean_root_sha256=H(temporac.prediction-root.v4, ASCII natural-clean, exact clean index bytes)",
        "prediction_drift_root_sha256=H(temporac.prediction-root.v4, ASCII natural-drift, exact drift index bytes)",
        "capability prediction_root_sha256 remains H(temporac.prediction-pair-root.v4, raw32(clean root), raw32(drift root))"
      ],
      "row_closed_keys": [
        "arm",
        "artifact_sha256",
        "component_key_hex",
        "condition",
        "opaque_key_hex",
        "population_row_sha256",
        "receipt_sha256",
        "seed",
        "slot",
        "split"
      ],
      "schema": "temporac.prediction-index.v4",
      "seeds": [
        20260815,
        20260816,
        20260817
      ],
      "top_level_closed_keys": [
        "condition",
        "contract_sha256",
        "population_manifest_sha256",
        "rows",
        "schema"
      ]
    },
    "target_index": {
      "consistency": "Each target artifact is unchanged seven-member temporac target NPZ; each receipt is unchanged ten-key temporac.target-receipt.v4; teacher digest is the selected artifact digest; no abstention target exists.",
      "inventory": "Every exact X0 target required by the proposal (3808 rows) plus every and only natural CERTIFIED row in the frozen K1 outcome index.",
      "order": "source_kind numeric, decoded source key bytes, numeric source_unit_index",
      "row_closed_keys": [
        "artifact_sha256",
        "receipt_sha256",
        "source_key_hex",
        "source_kind",
        "source_unit_index",
        "teacher_sha256"
      ],
      "schema": "temporac.target-index.v4",
      "top_level_closed_keys": [
        "contract_sha256",
        "k1_outcome_index_sha256",
        "rows",
        "schema"
      ]
    }
  },
  "cpu_replay_environment": {
    "cpython_initialization_profile": {
      "build_predicates": {
        "MS_WINDOWS": false,
        "PYMEM_ALLOCATOR_DEFAULT": 1,
        "PYMEM_ALLOCATOR_NOT_SET": 0,
        "Py_DEBUG": false,
        "Py_STATS": false,
        "WITH_PYMALLOC": true,
        "_PY_LONG_DEFAULT_MAX_STR_DIGITS": 4300,
        "_PyConfig_INIT_ISOLATED": 3,
        "__CYGWIN__": false,
        "platform": "linux-x86_64-little-endian-lp64"
      },
      "call_sequence": [
        "Before any CPython API call, native code validates exact launch-instance bytes, every exec-entry FD state, supervisor/initial ELF identity, cwd, argv and envp; it atomically closes every unlisted descriptor and records exec-entry readback.",
        "Still before any CPython API, set FD_CLOEXEC, pread/verify sealed inputs, close full-runtime fds 3/4/5, require retained/closed states and no extras, and record locked-pre-cpython readback.",
        "Call PySys_AddAuditHook exactly once and require return 0. Then call PyPreConfig_InitIsolatedConfig(&preconfig) exactly once; the supervisor makes no other CPython API call, explicit PyMem call, locale conversion or initialization call before/between these two ordered calls.",
        "Compare every Linux-applicable PyPreConfig field to constructor_defaults. Assign every final field in exact initconfig.h order; _config_init remains constructor-owned 3. No Windows-only field exists in the bound Linux build.",
        "Call Py_PreInitialize(&preconfig) exactly once and fail on PyStatus_Exception. No PyConfig setter, Py_DecodeLocale, supervisor PyMem allocation, PyConfig_Read, Py_Initialize, or other implicit preinitialization may precede this successful call.",
        "Call PyConfig_InitIsolatedConfig(&config) exactly once and compare every Linux-applicable field, including null pointers and zero-length/NULL-item lists, to pyconfig_constructor_defaults for the exact release build predicates.",
        "Traverse pyconfig_header_field_order exactly. Leave _config_init constructor-owned; assign each scalar directly; invoke PyConfig_SetString once for every owned string including NULL; invoke PyConfig_SetWideStringList once for every owned list including length-zero/NULL. Apply setter_semantics and check each PyStatus immediately.",
        "The supervisor does not call PyConfig_Read. Serialize the complete native pre-initialize struct snapshot in header order and require exact equality to the selected full-runtime or discovery final map.",
        "Call Py_InitializeFromConfig(&config) exactly once. Its bound internal configuration read may not change the selected map. On success obtain exact CPython-3.12.13 runtime PyPreConfig/PyConfig, call _Py_GetConfigsAsDict, and build temporac.cpython-initialization-readback.v4.",
        "Require all 10 preconfig fields, all 64 config fields, complete global-config object, sys argv/orig_argv/path/executable, filesystem/stream settings, and post-pyinitialize FD readback exact before any bootstrap import.",
        "Call PyConfig_Clear(&config) exactly once after successful readback and before bootstrap import. On every error after PyConfig_InitIsolatedConfig, call PyConfig_Clear exactly once before _exit(120); on Py_PreInitialize failure no PyConfig exists and no clear is called.",
        "After clear run only the selected bootstrap: discovery imports dataclasses as sole bootstrap action; full-runtime imports exact authority closure then dispatches. Joined normal exit calls Py_FinalizeEx exactly once; nonzero fails evidence."
      ],
      "cpython_source_bindings": [
        {
          "bytes": 54271,
          "path": "Doc/c-api/init_config.rst",
          "sha256": "b6064567d191ade7c873b229d8ded27b917710c0da367343edb13488d2c63d88"
        },
        {
          "bytes": 7820,
          "path": "Include/cpython/initconfig.h",
          "sha256": "86e3b9d1de6f310415912e2cdfdc276e311c026ec7fdf6190893f6313cd860a3"
        },
        {
          "bytes": 5706,
          "path": "Include/internal/pycore_initconfig.h",
          "sha256": "caf13e2c290ae8375636d0e1f3b1851a90396b3747da650d058c282b8743b558"
        },
        {
          "bytes": 62085,
          "path": "Lib/dataclasses.py",
          "sha256": "d242aea5fcf6408b1c1f622442f88f68b9526ce1f8bd2890d74a144677c427d9"
        },
        {
          "bytes": 92299,
          "path": "Python/initconfig.c",
          "sha256": "a50ff760d49e9d1da8a5a998bcedc8c377f12628d7d0426fc4dab89c1abfeff7"
        },
        {
          "bytes": 25540,
          "path": "Python/preconfig.c",
          "sha256": "837e6e39347c4ded7d053c7f23d3ac751614a8dc135332748af7a4b0fcd46cab"
        }
      ],
      "cpython_version": "3.12.13",
      "error_handling": {
        "error_exit_code": 120,
        "error_record": {
          "closed_keys": [
            "launch_instance_sha256",
            "schema",
            "stage"
          ],
          "rule": "The fail-closed native record contains no free-form CPython message, path, data, digest override, or alternate exit code; stage is one exact token and launch instance matches.",
          "schema": "temporac.native-initialization-error.v4"
        },
        "rule": "Every PyStatus is tested with PyStatus_Exception immediately; every void constructor/clear and native operation is followed by its specified structural/readback check. Only OK continues. ERROR, EXIT, nonzero return, errno mismatch, or readback mismatch records exactly one stage token, grants no authority, clears constructed PyConfig exactly once iff required, closes authority FDs, and _exit(120). Do not call Py_ExitStatusException, retry, fall back, finalize partial state, or reuse it.",
        "stage_tokens": [
          "EXEC_ENTRY_VALIDATE",
          "FD_LOCK_READBACK",
          "AUDIT_HOOK",
          "PRECONFIG_INIT",
          "PRECONFIG_COMPARE",
          "PY_PREINITIALIZE",
          "PYCONFIG_INIT",
          "PYCONFIG_CONSTRUCTOR_COMPARE",
          "PYCONFIG_SET_SCALAR",
          "PYCONFIG_SET_STRING",
          "PYCONFIG_SET_LIST",
          "PYCONFIG_PREINIT_READBACK",
          "PY_INITIALIZE_FROM_CONFIG",
          "PYCONFIG_RUNTIME_READBACK",
          "FD_POSTINIT_READBACK",
          "PYCONFIG_CLEAR",
          "BOOTSTRAP_IMPORT",
          "PY_FINALIZE"
        ]
      },
      "owned_list_field_order": [
        "orig_argv",
        "argv",
        "xoptions",
        "warnoptions",
        "module_search_paths"
      ],
      "owned_string_field_order": [
        "dump_refs_file",
        "filesystem_encoding",
        "filesystem_errors",
        "pycache_prefix",
        "stdio_encoding",
        "stdio_errors",
        "check_hash_pycs_mode",
        "program_name",
        "pythonpath_env",
        "home",
        "platlibdir",
        "stdlib_dir",
        "executable",
        "base_executable",
        "prefix",
        "base_prefix",
        "exec_prefix",
        "base_exec_prefix",
        "run_command",
        "run_module",
        "run_filename"
      ],
      "preconfig_constructor": "PyPreConfig_InitIsolatedConfig",
      "preconfig_constructor_defaults": {
        "_config_init": 3,
        "allocator": 0,
        "coerce_c_locale": 0,
        "coerce_c_locale_warn": 0,
        "configure_locale": 0,
        "dev_mode": 0,
        "isolated": 1,
        "parse_argv": 0,
        "use_environment": 0,
        "utf8_mode": 0
      },
      "preconfig_final_values": {
        "_config_init": 3,
        "allocator": 1,
        "coerce_c_locale": 0,
        "coerce_c_locale_warn": 0,
        "configure_locale": 0,
        "dev_mode": 0,
        "isolated": 1,
        "parse_argv": 0,
        "use_environment": 0,
        "utf8_mode": 0
      },
      "preconfig_header_field_order": [
        "_config_init",
        "parse_argv",
        "isolated",
        "use_environment",
        "configure_locale",
        "coerce_c_locale",
        "coerce_c_locale_warn",
        "utf8_mode",
        "dev_mode",
        "allocator"
      ],
      "profile_digest_rule": "The future temporac.cpython-initialization-profile.v4 artifact has exactly top_level_closed_keys, with contract_sha256 plus every literal/source/default/final/call/setter/readback/error rule in this record. Its SHA-256 covers complete canonical bytes including LF and is the sole cpython_initialization_profile_sha256 used by the environment, capsule, launcher, launch instance, and readbacks.",
      "pyconfig_constructor": "PyConfig_InitIsolatedConfig",
      "pyconfig_constructor_defaults": {
        "_config_init": 3,
        "_init_main": 1,
        "_install_importlib": 1,
        "_is_python_build": 0,
        "argv": [],
        "base_exec_prefix": null,
        "base_executable": null,
        "base_prefix": null,
        "buffered_stdio": 1,
        "bytes_warning": 0,
        "check_hash_pycs_mode": null,
        "code_debug_ranges": 1,
        "configure_c_stdio": 0,
        "dev_mode": 0,
        "dump_refs": 0,
        "dump_refs_file": null,
        "exec_prefix": null,
        "executable": null,
        "faulthandler": 0,
        "filesystem_encoding": null,
        "filesystem_errors": null,
        "hash_seed": 0,
        "home": null,
        "import_time": 0,
        "inspect": 0,
        "install_signal_handlers": 0,
        "int_max_str_digits": 4300,
        "interactive": 0,
        "isolated": 1,
        "malloc_stats": 0,
        "module_search_paths": [],
        "module_search_paths_set": 0,
        "optimization_level": 0,
        "orig_argv": [],
        "parse_argv": 0,
        "parser_debug": 0,
        "pathconfig_warnings": 0,
        "perf_profiling": 0,
        "platlibdir": null,
        "prefix": null,
        "program_name": null,
        "pycache_prefix": null,
        "pythonpath_env": null,
        "quiet": 0,
        "run_command": null,
        "run_filename": null,
        "run_module": null,
        "safe_path": 1,
        "show_ref_count": 0,
        "site_import": 1,
        "skip_source_first_line": 0,
        "stdio_encoding": null,
        "stdio_errors": null,
        "stdlib_dir": null,
        "tracemalloc": 0,
        "use_environment": 0,
        "use_frozen_modules": 1,
        "use_hash_seed": 0,
        "user_site_directory": 0,
        "verbose": 0,
        "warn_default_encoding": 0,
        "warnoptions": [],
        "write_bytecode": 1,
        "xoptions": []
      },
      "pyconfig_discovery_final_values": {
        "_config_init": 3,
        "_init_main": 1,
        "_install_importlib": 1,
        "_is_python_build": 0,
        "argv": [
          "temporac-generated-discovery"
        ],
        "base_exec_prefix": "/opt/temporac/replay-v4",
        "base_executable": "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
        "base_prefix": "/opt/temporac/replay-v4",
        "buffered_stdio": 1,
        "bytes_warning": 0,
        "check_hash_pycs_mode": "always",
        "code_debug_ranges": 1,
        "configure_c_stdio": 1,
        "dev_mode": 0,
        "dump_refs": 0,
        "dump_refs_file": null,
        "exec_prefix": "/opt/temporac/replay-v4",
        "executable": "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
        "faulthandler": 0,
        "filesystem_encoding": "utf-8",
        "filesystem_errors": "surrogateescape",
        "hash_seed": 0,
        "home": "/opt/temporac/replay-v4",
        "import_time": 0,
        "inspect": 0,
        "install_signal_handlers": 0,
        "int_max_str_digits": 4300,
        "interactive": 0,
        "isolated": 1,
        "malloc_stats": 0,
        "module_search_paths": [
          "/opt/temporac/replay-v4/lib/python3.12",
          "/opt/temporac/replay-v4/lib/python3.12/lib-dynload"
        ],
        "module_search_paths_set": 1,
        "optimization_level": 0,
        "orig_argv": [
          "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
          "--discover-generated-v4"
        ],
        "parse_argv": 0,
        "parser_debug": 0,
        "pathconfig_warnings": 0,
        "perf_profiling": 0,
        "platlibdir": "lib",
        "prefix": "/opt/temporac/replay-v4",
        "program_name": "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
        "pycache_prefix": null,
        "pythonpath_env": null,
        "quiet": 0,
        "run_command": null,
        "run_filename": null,
        "run_module": null,
        "safe_path": 1,
        "show_ref_count": 0,
        "site_import": 0,
        "skip_source_first_line": 0,
        "stdio_encoding": "utf-8",
        "stdio_errors": "surrogateescape",
        "stdlib_dir": "/opt/temporac/replay-v4/lib/python3.12",
        "tracemalloc": 0,
        "use_environment": 0,
        "use_frozen_modules": 0,
        "use_hash_seed": 1,
        "user_site_directory": 0,
        "verbose": 0,
        "warn_default_encoding": 0,
        "warnoptions": [],
        "write_bytecode": 0,
        "xoptions": []
      },
      "pyconfig_full_runtime_final_values": {
        "_config_init": 3,
        "_init_main": 1,
        "_install_importlib": 1,
        "_is_python_build": 0,
        "argv": [
          "temporac-origin-supervisor"
        ],
        "base_exec_prefix": "/opt/temporac/replay-v4",
        "base_executable": "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
        "base_prefix": "/opt/temporac/replay-v4",
        "buffered_stdio": 1,
        "bytes_warning": 0,
        "check_hash_pycs_mode": "always",
        "code_debug_ranges": 1,
        "configure_c_stdio": 1,
        "dev_mode": 0,
        "dump_refs": 0,
        "dump_refs_file": null,
        "exec_prefix": "/opt/temporac/replay-v4",
        "executable": "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
        "faulthandler": 0,
        "filesystem_encoding": "utf-8",
        "filesystem_errors": "surrogateescape",
        "hash_seed": 0,
        "home": "/opt/temporac/replay-v4",
        "import_time": 0,
        "inspect": 0,
        "install_signal_handlers": 0,
        "int_max_str_digits": 4300,
        "interactive": 0,
        "isolated": 1,
        "malloc_stats": 0,
        "module_search_paths": [
          "/opt/temporac/replay-v4/lib/python3.12",
          "/opt/temporac/replay-v4/lib/python3.12/lib-dynload",
          "/opt/temporac/replay-v4/lib/python3.12/site-packages"
        ],
        "module_search_paths_set": 1,
        "optimization_level": 0,
        "orig_argv": [
          "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
          "--embedded-cpython-v4"
        ],
        "parse_argv": 0,
        "parser_debug": 0,
        "pathconfig_warnings": 0,
        "perf_profiling": 0,
        "platlibdir": "lib",
        "prefix": "/opt/temporac/replay-v4",
        "program_name": "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
        "pycache_prefix": null,
        "pythonpath_env": null,
        "quiet": 0,
        "run_command": null,
        "run_filename": null,
        "run_module": null,
        "safe_path": 1,
        "show_ref_count": 0,
        "site_import": 0,
        "skip_source_first_line": 0,
        "stdio_encoding": "utf-8",
        "stdio_errors": "surrogateescape",
        "stdlib_dir": "/opt/temporac/replay-v4/lib/python3.12",
        "tracemalloc": 0,
        "use_environment": 0,
        "use_frozen_modules": 0,
        "use_hash_seed": 1,
        "user_site_directory": 0,
        "verbose": 0,
        "warn_default_encoding": 0,
        "warnoptions": [],
        "write_bytecode": 0,
        "xoptions": []
      },
      "pyconfig_header_field_order": [
        "_config_init",
        "isolated",
        "use_environment",
        "dev_mode",
        "install_signal_handlers",
        "use_hash_seed",
        "hash_seed",
        "faulthandler",
        "tracemalloc",
        "perf_profiling",
        "import_time",
        "code_debug_ranges",
        "show_ref_count",
        "dump_refs",
        "dump_refs_file",
        "malloc_stats",
        "filesystem_encoding",
        "filesystem_errors",
        "pycache_prefix",
        "parse_argv",
        "orig_argv",
        "argv",
        "xoptions",
        "warnoptions",
        "site_import",
        "bytes_warning",
        "warn_default_encoding",
        "inspect",
        "interactive",
        "optimization_level",
        "parser_debug",
        "write_bytecode",
        "verbose",
        "quiet",
        "user_site_directory",
        "configure_c_stdio",
        "buffered_stdio",
        "stdio_encoding",
        "stdio_errors",
        "check_hash_pycs_mode",
        "use_frozen_modules",
        "safe_path",
        "int_max_str_digits",
        "pathconfig_warnings",
        "program_name",
        "pythonpath_env",
        "home",
        "platlibdir",
        "module_search_paths_set",
        "module_search_paths",
        "stdlib_dir",
        "executable",
        "base_executable",
        "prefix",
        "base_prefix",
        "exec_prefix",
        "base_exec_prefix",
        "skip_source_first_line",
        "run_command",
        "run_module",
        "run_filename",
        "_install_importlib",
        "_init_main",
        "_is_python_build"
      ],
      "pyconfig_read_rule": "The supervisor never calls public PyConfig_Read. The internal CPython-3.12.13 configuration read necessarily performed inside Py_InitializeFromConfig is part of the bound interpreter implementation; because every input/output/path/default field is explicit, its complete post-init 64-field result must remain exactly the selected final map. Any implicit default fill, argv parse, environment/locale/path discovery, or field drift is a failure.",
      "readback": {
        "closed_keys": [
          "config",
          "global_config_bytes",
          "global_config_sha256",
          "pre_config",
          "schema",
          "stream_and_sys"
        ],
        "config_closed_keys": [
          "_config_init",
          "isolated",
          "use_environment",
          "dev_mode",
          "install_signal_handlers",
          "use_hash_seed",
          "hash_seed",
          "faulthandler",
          "tracemalloc",
          "perf_profiling",
          "import_time",
          "code_debug_ranges",
          "show_ref_count",
          "dump_refs",
          "dump_refs_file",
          "malloc_stats",
          "filesystem_encoding",
          "filesystem_errors",
          "pycache_prefix",
          "parse_argv",
          "orig_argv",
          "argv",
          "xoptions",
          "warnoptions",
          "site_import",
          "bytes_warning",
          "warn_default_encoding",
          "inspect",
          "interactive",
          "optimization_level",
          "parser_debug",
          "write_bytecode",
          "verbose",
          "quiet",
          "user_site_directory",
          "configure_c_stdio",
          "buffered_stdio",
          "stdio_encoding",
          "stdio_errors",
          "check_hash_pycs_mode",
          "use_frozen_modules",
          "safe_path",
          "int_max_str_digits",
          "pathconfig_warnings",
          "program_name",
          "pythonpath_env",
          "home",
          "platlibdir",
          "module_search_paths_set",
          "module_search_paths",
          "stdlib_dir",
          "executable",
          "base_executable",
          "prefix",
          "base_prefix",
          "exec_prefix",
          "base_exec_prefix",
          "skip_source_first_line",
          "run_command",
          "run_module",
          "run_filename",
          "_install_importlib",
          "_init_main",
          "_is_python_build"
        ],
        "pre_config_closed_keys": [
          "_config_init",
          "parse_argv",
          "isolated",
          "use_environment",
          "configure_locale",
          "coerce_c_locale",
          "coerce_c_locale_warn",
          "utf8_mode",
          "dev_mode",
          "allocator"
        ],
        "rule": "Native access to the exact 3.12.13 runtime structs verifies all header fields, including dump_refs_file which _PyConfig_AsDict omits. The complete _Py_GetConfigsAsDict global_config object is separately canonicalized with LF and hashed. Unknown/missing field, pointer/list alias, normalized value drift, or mismatch fails before import.",
        "schema": "temporac.cpython-initialization-readback.v4",
        "stream_and_sys_closed_keys": [
          "filesystem_encoding",
          "filesystem_errors",
          "stderr_encoding",
          "stderr_errors",
          "stdin_encoding",
          "stdin_errors",
          "stdout_encoding",
          "stdout_errors",
          "sys_argv",
          "sys_executable",
          "sys_orig_argv",
          "sys_path"
        ],
        "stream_rule": "sys.argv/orig_argv/path/executable and filesystem/stdio values are exact selected-profile projections; stdin/stdout errors equal surrogateescape and stderr errors equals backslashreplace. All encodings normalize to utf-8."
      },
      "scalar_field_order": [
        "isolated",
        "use_environment",
        "dev_mode",
        "install_signal_handlers",
        "use_hash_seed",
        "hash_seed",
        "faulthandler",
        "tracemalloc",
        "perf_profiling",
        "import_time",
        "code_debug_ranges",
        "show_ref_count",
        "dump_refs",
        "malloc_stats",
        "parse_argv",
        "site_import",
        "bytes_warning",
        "warn_default_encoding",
        "inspect",
        "interactive",
        "optimization_level",
        "parser_debug",
        "write_bytecode",
        "verbose",
        "quiet",
        "user_site_directory",
        "configure_c_stdio",
        "buffered_stdio",
        "use_frozen_modules",
        "safe_path",
        "int_max_str_digits",
        "pathconfig_warnings",
        "module_search_paths_set",
        "skip_source_first_line",
        "_install_importlib",
        "_init_main",
        "_is_python_build"
      ],
      "schema": "temporac.cpython-initialization-profile.v4",
      "setter_semantics": {
        "list_rule": "For each of orig_argv, argv, xoptions, warnoptions, module_search_paths in initconfig.h order, construct a native wchar_t array from the exact ASCII literals by one code-point-to-wchar_t widening; call PyConfig_SetWideStringList exactly once. Empty lists use length 0 and items NULL. The setter's internal _Py_PreInitializeFromConfig call must observe the already successful identical preconfiguration and must not alter it.",
        "scalar_rule": "Assign each scalar/unsigned-long field directly once in initconfig.h order using the exact JSON integer. No sentinel remains. _config_init stays the constructor value 3 and is checked but not assigned.",
        "string_rule": "For every owned wchar_t pointer in initconfig.h order, call PyConfig_SetString exactly once. ASCII values widen one byte to one equal wchar_t code point; JSON null passes a NULL pointer. PyConfig_SetBytesString, PyConfig_SetBytesArgv, PyConfig_SetArgv, Py_DecodeLocale, locale conversion, caller buffers, and implicit path discovery are forbidden.",
        "verification": "After every setter, check its PyStatus immediately; after the complete traversal require pointer ownership, nullness, list length/items, and all 64 field values exactly equal the selected final map before Py_InitializeFromConfig."
      },
      "top_level_closed_keys": [
        "build_predicates",
        "call_sequence",
        "contract_sha256",
        "cpython_source_bindings",
        "cpython_version",
        "error_handling",
        "owned_list_field_order",
        "owned_string_field_order",
        "preconfig_constructor",
        "preconfig_constructor_defaults",
        "preconfig_final_values",
        "preconfig_header_field_order",
        "profile_digest_rule",
        "pyconfig_constructor",
        "pyconfig_constructor_defaults",
        "pyconfig_discovery_final_values",
        "pyconfig_full_runtime_final_values",
        "pyconfig_header_field_order",
        "pyconfig_read_rule",
        "readback",
        "scalar_field_order",
        "schema",
        "setter_semantics"
      ]
    },
    "cpython_runtime_member_index": {
      "completeness": [
        "Under the exact stdlib root, include every regular non-symlink .py file recursively as stdlib-source. Under exact lib-dynload, include every regular file whose suffix is one exact importlib.machinery.EXTENSION_SUFFIXES value as stdlib-extension. Files are enumerated without import, glob ambiguity, or locale collation. No pyc, __pycache__, stdlib zip, user-site, vendor overlay, symlink, hardlink/case alias, or path outside the two roots is permitted.",
        "builtin_module_names is the duplicate-free ASCII sort of sys.builtin_module_names from the bound interpreter. frozen_module_names is the duplicate-free ASCII sort returned by exact CPython-3.12 _imp._frozen_module_names(); absence or disagreement with FrozenImporter origin checks fails. Each name has one builtin-provider or frozen-provider row whose bytes/hash are the exact libpython provider image named by provider_path.",
        "Every stdlib extension row is byte-identical to the same runtime-relative object in the ELF index. Every imported stdlib source/extension/builtin/frozen origin must resolve to exactly one row. Extra or missing source/extension file, module name, row, or executed origin fails."
      ],
      "kind_tokens": [
        "builtin-provider",
        "frozen-provider",
        "stdlib-extension",
        "stdlib-source"
      ],
      "normalization": "normalized_path is stdlib/<relative .py>, lib-dynload/<relative extension>, builtin/<ASCII module>, or frozen/<ASCII module>; provider_path is exact runtime-root-relative POSIX path. Both reject absolute, dot, alternate separator, non-NFC, duplicate, symlink, hardlink/case alias, or traversal.",
      "order": "ASCII kind then normalized_path then provider_path",
      "row_closed_keys": [
        "bytes",
        "kind",
        "normalized_path",
        "provider_path",
        "sha256"
      ],
      "schema": "temporac.cpython-runtime-member-index.v4",
      "top_level_closed_keys": [
        "builtin_module_names",
        "contract_sha256",
        "extension_suffixes",
        "frozen_module_names",
        "interpreter_image_sha256",
        "libpython_image_sha256",
        "rows",
        "schema"
      ]
    },
    "digest_binding": "environment_sha256 hashes complete exact environment bytes, including floating-point control, CPython initialization profile, FD contract (which contains the memfd procedure and readback schemas), native launcher, discovery capsule/precommit/two-run evidence-index digest/root/output/trusted-source digests, and installed-distribution/CPython/wheel/ELF indexes; it agrees in every checkpoint, tune, selection, G1, G5a and K7 object.",
    "elf_index": {
      "closed_keys": [
        "contract_sha256",
        "rows",
        "schema"
      ],
      "note": "Paths are evidence inside the sealed environment lock and are not substituted for bytes or hashes.",
      "order": "ASCII normalized absolute loader path",
      "row_closed_keys": [
        "build_id_or_null",
        "bytes",
        "path",
        "sha256",
        "soname"
      ],
      "schema": "temporac.cpu-elf-index.v4"
    },
    "fd_contract": {
      "constant_ledger": {
        "required_symbols": [
          "F_ADD_SEALS",
          "F_DUPFD_CLOEXEC",
          "F_GETFD",
          "F_GETFL",
          "F_GETPIPE_SZ",
          "F_GET_SEALS",
          "F_SEAL_GROW",
          "F_SEAL_SEAL",
          "F_SEAL_SHRINK",
          "F_SEAL_WRITE",
          "FD_CLOEXEC",
          "MFD_ALLOW_SEALING",
          "MFD_CLOEXEC",
          "O_ACCMODE",
          "O_APPEND",
          "O_ASYNC",
          "O_CLOEXEC",
          "O_DIRECT",
          "O_DSYNC",
          "O_LARGEFILE",
          "O_NONBLOCK",
          "O_PATH",
          "O_RDONLY",
          "O_RDWR",
          "O_SYNC",
          "O_WRONLY"
        ],
        "rule": "temporac.linux-fd-constant-ledger.v4 records the exact nonnegative Linux x86_64 numeric value for every symbol, exact kernel/libc header bytes/hash, and schema. Raw descriptor/status/seal values must equal the stated bitwise construction from this ledger; recording an observed extra bit does not authorize it.",
        "schema": "temporac.linux-fd-constant-ledger.v4"
      },
      "discovery_layout": [
        {
          "access_mode": "read-only",
          "fd": 0,
          "object_type": "sealed-memfd",
          "role": "discovery-capsule"
        },
        {
          "access_mode": "write-only",
          "fd": 1,
          "object_type": "char-device-1:3",
          "role": "stdout-null"
        },
        {
          "access_mode": "write-only",
          "fd": 2,
          "object_type": "char-device-1:3",
          "role": "stderr-null"
        },
        {
          "access_mode": "write-only",
          "fd": 3,
          "object_type": "pipe-write-end",
          "role": "discovery-output"
        }
      ],
      "fd_row_closed_keys": [
        "access_mode",
        "descriptor_flags_entry_u32",
        "descriptor_flags_locked_u32",
        "device_major_or_null",
        "device_minor_or_null",
        "eof_rule",
        "fd",
        "file_status_flags_u32",
        "inheritable_after_lock",
        "inheritable_at_exec_entry",
        "initial_offset_or_null",
        "object_type",
        "payload_bytes_or_null",
        "payload_sha256_or_null",
        "pipe_capacity_bytes_or_null",
        "queued_bytes_or_null",
        "role",
        "seals",
        "seekable",
        "size_bytes_or_null"
      ],
      "full_runtime_layout": [
        {
          "access_mode": "read-only",
          "fd": 0,
          "object_type": "sealed-memfd",
          "role": "invocation"
        },
        {
          "access_mode": "write-only",
          "fd": 1,
          "object_type": "char-device-1:3",
          "role": "stdout-null"
        },
        {
          "access_mode": "write-only",
          "fd": 2,
          "object_type": "char-device-1:3",
          "role": "stderr-null"
        },
        {
          "access_mode": "read-only",
          "fd": 3,
          "object_type": "sealed-memfd",
          "role": "effective-contract"
        },
        {
          "access_mode": "read-only",
          "fd": 4,
          "object_type": "sealed-memfd",
          "role": "source-allowlist"
        },
        {
          "access_mode": "read-only",
          "fd": 5,
          "object_type": "sealed-memfd",
          "role": "cpu-environment"
        },
        {
          "access_mode": "write-only",
          "fd": 6,
          "object_type": "pipe-write-end",
          "role": "origin-trace-output"
        }
      ],
      "launch_instance": {
        "closed_keys": [
          "argv",
          "contract_sha256",
          "cwd",
          "environment_sha256_or_null",
          "fd_rows",
          "launcher_sha256",
          "mode",
          "schema",
          "sealed_memfd_construction_index_bytes_hex",
          "sealed_memfd_construction_index_sha256"
        ],
        "digest_rule": "launch_instance_sha256 is SHA-256 of the complete canonical launch-instance bytes including LF. The CPU environment contains only this closed schema/profile, never an invocation-specific payload digest; each post-environment launch instance may bind environment_sha256 without a self-cycle.",
        "mode_tokens": [
          "bootstrap-generated-discovery",
          "full-runtime"
        ],
        "mode_values": {
          "bootstrap-generated-discovery": "argv equals the exact discovery capsule argv, cwd '/', environment_sha256_or_null is null, fd_rows are exactly discovery_layout 0..3, the construction index contains exactly discovery fd0, and launcher/contract/capsule payloads equal the capsule/precommit",
          "full-runtime": "argv equals runtime_launcher.argv, cwd '/', environment_sha256_or_null is the exact final environment digest, fd_rows are exactly full_runtime_layout 0..6, the construction index contains exactly fds 0,3,4,5 in fd order, and launcher/contract/source/environment payloads match"
        },
        "schema": "temporac.native-launch-instance.v4",
        "sealed_memfd_binding": "sealed_memfd_construction_index_bytes_hex is lowercase even-length hex decoding to the complete canonical temporac.sealed-memfd-construction-index.v4 bytes including LF; its digest is exact. Each index row embeds one complete construction-readback object as lowercase bytes hex plus digest. The launch-instance hash therefore commits every transient construction step without changing the existing 20-key launch-FD or 15-key child-readback rows."
      },
      "readback": {
        "closed_keys": [
          "extra_fds",
          "launch_instance_sha256",
          "rows",
          "schema",
          "stage"
        ],
        "extra_rule": "extra_fds is exactly [] at every committed boundary. The single /proc/self/fd enumeration descriptor is excluded only while enumerating itself, is closed before serialization, and cannot be reused or retained.",
        "row_closed_keys": [
          "access_mode_or_null",
          "descriptor_flags_u32_or_null",
          "device_major_or_null",
          "device_minor_or_null",
          "fd",
          "file_status_flags_u32_or_null",
          "inheritable_or_null",
          "object_type_or_null",
          "offset_or_null",
          "pipe_capacity_bytes_or_null",
          "queued_bytes_or_null",
          "seals_or_null",
          "seekable_or_null",
          "size_bytes_or_null",
          "state"
        ],
        "row_rule": "Rows cover every numeric fd from the mode layout in ascending order. state is present or closed. A present row repeats a fresh fstat/F_GETFD/F_GETFL/lseek-or-ESPIPE/F_GET_SEALS/F_GETPIPE_SZ/FIONREAD projection and exactly equals the applicable launch row; no *_or_null is null unless inapplicable by object type. A closed row has every *_or_null null and must return EBADF. At exec-entry every layout fd is present with descriptor flags zero. At both later stages full-runtime fds 3,4,5 are closed and all other layout fds are present/locked; discovery keeps all 0..3 present/locked.",
        "schema": "temporac.linux-fd-readback.v4",
        "stage_tokens": [
          "exec-entry",
          "locked-pre-cpython",
          "post-pyinitialize"
        ]
      },
      "row_semantics": [
        "At exec-entry descriptor_flags_entry_u32 is exactly zero, hence every listed descriptor is inheritable_at_exec_entry=true. After the first native snapshot set FD_CLOEXEC and require descriptor_flags_locked_u32 exactly FD_CLOEXEC and inheritable_after_lock=false.",
        "A sealed-memfd follows sealed_memfd_construction exactly: keep the writable writer open through populate/verify/seal and through open(/proc/self/fd/<writer>,O_RDONLY|O_CLOEXEC); prove the reader is a distinct open-file description for the same inode, then close/prove-EBADF the writer, verify the surviving reader again, and only then dup3 it with flags zero to the prescribed child fd. At exec, file_status_flags_u32 equals O_RDONLY|O_LARGEFILE; seekable=true; initial offset=0; size=payload_bytes; payload hash covers exactly size bytes; seals are exactly the four value-domain tokens; pread(size) then pread(one byte at size) proves exact EOF while offset remains zero.",
        "FD1 and FD2 are independently opened /dev/null character devices with major 1 minor 3, O_WRONLY|O_LARGEFILE, seekable=true, offset zero, st_size zero, no seals/payload/pipe fields, and EOF token not-applicable-character-sink. Sharing or duplicating one open-file description is forbidden.",
        "A pipe-write-end has O_WRONLY and no append/nonblock/async/direct/sync/path bit; seekable=false with ESPIPE and null offset/size/seals/payload/device fields; exact positive F_GETPIPE_SZ, FIONREAD queued bytes zero at entry, exactly one controller-held read peer, and EOF token joined-reader-eof-after-final-writer-close.",
        "The controller closes every fd outside the exact mode layout before absolute-path exec. The supervisor records exec-entry readback, sets FD_CLOEXEC, preads and validates sealed inputs, closes full-runtime fds3/4/5, and records locked-pre-cpython readback. It retains full-runtime 0/1/2/6 and discovery 0/1/2/3. Immediately after Py_InitializeFromConfig it records post-pyinitialize readback; retained state must be unchanged and every required closed fd still EBADF.",
        "Each child exec constructs a fresh launch instance and descriptor table. No inherited duplicate, fork-without-exec, caller fd, socket, tty, disk regular file, writable input memfd, shared open-file description, changed offset/size/seal/flag, early EOF, unread output peer, extra fd, or missing readback is permitted."
      ],
      "schema": "temporac.linux-fd-contract.v4",
      "sealed_memfd_construction": {
        "applicable_rows": "Exactly discovery fd0 and full-runtime fds 0,3,4,5. Every payload is positive exact canonical bytes already named by its launch row. No other descriptor or payload may use this procedure.",
        "construction_readback": {
          "closed_keys": [
            "child_fd",
            "mode",
            "payload_bytes",
            "payload_sha256",
            "reader_fd",
            "role",
            "rows",
            "schema",
            "writer_fd"
          ],
          "digest_rule": "construction_readback_sha256 is SHA-256 of the exact standalone canonical temporac.sealed-memfd-construction-readback.v4 bytes including LF. It is retained with the controller launch evidence; every stated syscall return, errno, fstat/fcntl/lseek/pread value and step transition is recomputed from the exact process before exec. It is not an environment input and cannot authorize bytes.",
          "index": {
            "row_closed_keys": [
              "child_fd",
              "readback_bytes_hex",
              "readback_sha256",
              "role"
            ],
            "row_order": "numeric child_fd ascending and exactly the mode's applicable sealed rows",
            "schema": "temporac.sealed-memfd-construction-index.v4",
            "top_level_closed_keys": [
              "mode",
              "rows",
              "schema"
            ],
            "validation": "readback_bytes_hex is lowercase even-length hex of the complete exact standalone readback bytes including LF; digest, embedded child_fd/mode/role, payload and stage rows all match. Missing, extra, reordered, duplicate, digest-only, alternate-encoded or cross-launch readback fails."
          },
          "row_closed_keys": [
            "child_descriptor_flags_u32_or_null",
            "child_status_flags_u32_or_null",
            "reader_descriptor_flags_u32_or_null",
            "reader_offset_or_null",
            "reader_status_flags_u32_or_null",
            "same_device_or_null",
            "same_inode_or_null",
            "same_open_file_description_or_null",
            "seals",
            "stage",
            "writer_descriptor_flags_u32_or_null",
            "writer_offset_or_null",
            "writer_status_flags_u32_or_null"
          ],
          "row_order": "exact stage_tokens order; one row per token and no extra row",
          "schema": "temporac.sealed-memfd-construction-readback.v4",
          "stage_tokens": [
            "writer-created",
            "writer-populated-verified",
            "writer-sealed",
            "reader-opened",
            "ofd-independence-proved",
            "writer-closed",
            "reader-postclose-verified",
            "child-fd-installed"
          ],
          "stage_value_rules": {
            "child-fd-installed": "child descriptor/status/offset are exact zero, O_RDONLY|O_LARGEFILE and zero; reader and writer fields are null; seals is the exact four-token list; same_device=true, same_inode=true and same_open_file_description=true compare child to the immediately pre-close reader snapshot",
            "ofd-independence-proved": "writer and reader descriptor/status/offset are FD_CLOEXEC, writer O_RDWR|O_LARGEFILE offset zero and reader O_RDONLY|O_LARGEFILE offset zero after the sole lseek proof; child fields are null; seals is the exact four-token list; same_device=true, same_inode=true and same_open_file_description=false",
            "reader-opened": "writer and reader descriptor/status/offset are FD_CLOEXEC, writer O_RDWR|O_LARGEFILE offset zero and reader O_RDONLY|O_LARGEFILE offset zero; child fields are null; seals is the exact four-token list; all same_* fields are null until the ordered comparison/proof stage",
            "reader-postclose-verified": "reader descriptor/status/offset are FD_CLOEXEC, O_RDONLY|O_LARGEFILE and zero; writer and child fields are null; seals is the exact four-token list; same_device=true and same_inode=true compare reader to captured writer identity; same_open_file_description=false retains the proved writer-reader relation",
            "writer-closed": "reader descriptor/status/offset remain FD_CLOEXEC, O_RDONLY|O_LARGEFILE and zero; writer and child fields are null after close plus EBADF; seals is the exact four-token list; same_device=true, same_inode=true and same_open_file_description=false retain the verified writer-reader relation",
            "writer-created": "writer descriptor/status/offset are FD_CLOEXEC, O_RDWR|O_LARGEFILE and zero; reader and child fields and all same_* fields are null; seals is []",
            "writer-populated-verified": "the same presence and exact writer descriptor/status/offset values as writer-created; reader, child and same_* fields remain null; seals is []; exact payload_bytes, payload_sha256, size and EOF checks have succeeded",
            "writer-sealed": "writer descriptor/status/offset remain FD_CLOEXEC, O_RDWR|O_LARGEFILE and zero; reader, child and same_* fields are null; seals is exactly [F_SEAL_GROW,F_SEAL_SEAL,F_SEAL_SHRINK,F_SEAL_WRITE]"
          }
        },
        "error_contract": {
          "closed_keys": [
            "child_fd",
            "errno",
            "mode",
            "role",
            "schema",
            "stage"
          ],
          "errno_rule": "For a syscall/fcntl/lseek/pread/pwrite/fstat failure, errno is the exact positive Linux errno captured immediately before any other call; EINTR is recorded and never retried. For an unexpected successful return, short operation, byte/hash/value mismatch, wrong null/presence state, or other logical verification failure with no failing system call, errno is exact integer zero. The stage is the first failing operation's sole token; alternate sentinel, stale errno, negative value, omitted row or second error row fails.",
          "rule": "Any syscall error, EINTR, short read/write, wrong return, errno/readback mismatch or unexpected descriptor state emits exactly one canonical error row, closes every raw/transient/child descriptor created for this row, creates no launch instance or discovery evidence, and terminates the controller failure path. errno is exactly the value defined by errno_rule; success never emits this object.",
          "schema": "temporac.sealed-memfd-construction-error.v4",
          "stage_tokens": [
            "MEMFD_CREATE",
            "MEMFD_PROMOTE_WRITER",
            "MEMFD_TRUNCATE",
            "MEMFD_WRITE",
            "MEMFD_WRITER_VERIFY",
            "MEMFD_SEAL",
            "MEMFD_OPEN_READER",
            "MEMFD_PROMOTE_READER",
            "MEMFD_READER_VERIFY",
            "MEMFD_OFFSET_INDEPENDENCE",
            "MEMFD_CLOSE_WRITER",
            "MEMFD_POSTCLOSE_VERIFY",
            "MEMFD_DUP_CHILD",
            "MEMFD_FINAL_VERIFY"
          ]
        },
        "memfd_name_rule": "The ASCII memfd name is exactly temporac.<mode>.<role>.v4 using the launch-instance mode and row role tokens; no NUL, truncation, caller suffix or alternate name. Its procfs symlink is never an authority path.",
        "operation_order": [
          "Call memfd_create(exact memfd_name,MFD_ALLOW_SEALING|MFD_CLOEXEC) once. Immediately promote the returned raw writer with fcntl(F_DUPFD_CLOEXEC,64), require writer_fd>=64, close the raw descriptor and require F_GETFD on it returns -1/EBADF. The surviving writer has FD_CLOEXEC, access O_RDWR and exact status O_RDWR|O_LARGEFILE.",
          "Call ftruncate(writer,payload_bytes) once, then one pwrite(writer,exact payload,payload_bytes,0); require the exact positive length, no retry or short write. Require writer offset zero, fstat regular anonymous file with exact size, F_GET_SEALS zero, one pread of exact size byte-equal to payload, one pread at exact size returns zero, and recomputed SHA-256 equals payload_sha256.",
          "While writer remains open, call fcntl(writer,F_ADD_SEALS,F_SEAL_GROW|F_SEAL_SEAL|F_SEAL_SHRINK|F_SEAL_WRITE) once and require return zero; F_GET_SEALS must equal exactly that mask. No writable mapping or second writer exists.",
          "Form exact ASCII /proc/self/fd/<base10 writer_fd> and call open(path,O_RDONLY|O_CLOEXEC) once while writer is still open. Promote the raw reader with fcntl(F_DUPFD_CLOEXEC,64), require reader_fd>=64 and reader_fd!=writer_fd, close the raw descriptor and require it EBADF. The surviving reader has FD_CLOEXEC and exact status O_RDONLY|O_LARGEFILE.",
          "With both writer and reader open, require fstat st_dev/st_ino/regular type/size identical, exact seals on both, both offsets zero, and exact reader payload/hash/EOF by pread. Prove distinct open-file descriptions in the sole order: lseek(reader,1,SEEK_SET)==1; lseek(writer,0,SEEK_CUR)==0; lseek(reader,0,SEEK_SET)==0; then both current offsets are zero. A dup/shared-offset reader fails.",
          "Close writer exactly once and require subsequent fcntl(writer,F_GETFD) returns -1 with errno EBADF. With only reader open, repeat fstat type/device/inode/size, F_GETFL, F_GETFD, F_GET_SEALS, offset-zero, exact payload/hash and EOF checks; any drift fails.",
          "Call dup3(reader,child_fd,0) exactly once, require return child_fd, then close reader and require it EBADF. The child descriptor has descriptor flags zero, exact O_RDONLY|O_LARGEFILE status, offset zero, identical device/inode/size/seals and exact payload/hash/EOF. This is the only descriptor passed to exec for that payload.",
          "Serialize all eight construction readback rows before exec. Transient raw/writer/reader fds are absent from the child table; exact launch layout and exec-entry readback remain the existing 20-key/15-key contracts. Reopen after writer close, dup of writer, writable child input, disk fallback, mmap, alternate proc path, shared OFD, missing step or reordered step fails."
        ],
        "schema": "temporac.sealed-memfd-construction-procedure.v4"
      },
      "value_domains": {
        "access_mode_tokens": [
          "read-only",
          "write-only"
        ],
        "eof_rule_tokens": [
          "joined-reader-eof-after-final-writer-close",
          "not-applicable-character-sink",
          "sealed-exact-size-then-eof"
        ],
        "object_type_tokens": [
          "char-device-1:3",
          "pipe-write-end",
          "sealed-memfd"
        ],
        "seals_rule": "seals is [] for char-device/pipe and exactly ASCII-ordered [F_SEAL_GROW,F_SEAL_SEAL,F_SEAL_SHRINK,F_SEAL_WRITE] for sealed-memfd; no integer-only or unordered substitute",
        "types": "fd and every *_u32/*_bytes/device/offset integer are JSON integers in the specified nonnegative range or exact null; booleans are JSON booleans; hashes lowercase 64-hex; role/object/access/EOF values are exact tokens"
      }
    },
    "field_schemas": {
      "architecture": {
        "closed_keys": [
          "byteorder",
          "machine",
          "platform_tag",
          "pointer_bits"
        ],
        "types": "platform_tag is nonempty ASCII; byteorder is exact little, machine exact x86_64, pointer_bits exact integer 64"
      },
      "autocast": {
        "closed_keys": [
          "cpu_dtype",
          "cpu_enabled",
          "cuda_dtype",
          "cuda_enabled"
        ],
        "types": "dtype names are nonempty ASCII and enabled fields are booleans; both enabled values must be false"
      },
      "blas": {
        "closed_keys": [
          "configuration_sha256",
          "library_sha256",
          "threading_layer",
          "vendor",
          "version"
        ],
        "types": "hashes bind exact configuration bytes and loaded library bytes; other fields are nonempty ASCII"
      },
      "cpu_dispatch": {
        "closed_keys": [
          "numpy_baseline",
          "numpy_found",
          "numpy_not_found",
          "onednn_primitive_isa",
          "torch_cpu_capability",
          "torch_dispatch_sha256"
        ],
        "types": "ISA lists are ASCII arrays in provider order; digest binds exact Torch dispatch report bytes; scalar values are nonempty ASCII"
      },
      "cpu_isa": {
        "closed_keys": [
          "cpu_model",
          "flags_sha256",
          "microcode_sha256",
          "ordered_flags"
        ],
        "types": "model is nonempty ASCII, ordered_flags is a duplicate-free ASCII array in probe order, and hashes bind exact sanitized probe bytes"
      },
      "determinism": {
        "closed_keys": [
          "cudnn_allow_tf32",
          "cudnn_benchmark",
          "cudnn_deterministic",
          "cuda_matmul_allow_tf32",
          "matmul_precision",
          "pythonhashseed",
          "torch_deterministic_algorithms",
          "torch_deterministic_warn_only"
        ],
        "types": "flags are booleans, matmul_precision is exact highest, pythonhashseed is exact ASCII 0; deterministic algorithms true, warn_only false, both TF32 flags false"
      },
      "environment_variables": {
        "closed_keys": [
          "BLIS_NUM_THREADS",
          "MKL_DYNAMIC",
          "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS",
          "OMP_DYNAMIC",
          "OMP_NUM_THREADS",
          "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS"
        ],
        "types": "exact ASCII values; all thread counts are 1 and both dynamic flags are FALSE"
      },
      "floating_point_control": {
        "closed_keys": [
          "backend_state",
          "fe_rounding",
          "helper",
          "mxcsr",
          "probe",
          "torch_flush_denormal"
        ],
        "types": "all records are closed objects below; every requested set/read/probe succeeds exactly or replay fails closed"
      },
      "numpy": {
        "closed_keys": [
          "configuration_sha256",
          "version",
          "wheel_sha256"
        ],
        "types": "version is nonempty ASCII; hashes bind exact provider report and exact wheel-index row"
      },
      "onednn": {
        "closed_keys": [
          "build_sha256",
          "enabled",
          "primitive_isa",
          "version"
        ],
        "types": "enabled is boolean; other fields are nonempty ASCII or exact build-report digest"
      },
      "python": {
        "closed_keys": [
          "abi_tag",
          "build",
          "compiler",
          "executable_sha256",
          "implementation",
          "version"
        ],
        "types": "five nonempty ASCII strings and exact interpreter-file digest"
      },
      "runtime_launcher": {
        "closed_keys": [
          "argv",
          "cpython_initialization_profile_sha256",
          "envp",
          "executable_bytes",
          "executable_path",
          "executable_sha256",
          "fd_contract_sha256",
          "launch_instance_schema",
          "module_search_paths",
          "preimport_sequence",
          "profile_name",
          "schema",
          "working_directory"
        ],
        "types": "all literal fields match runtime_launcher; future positive byte counts/hashes name the exact regular supervisor and exact initialization/FD profile artifacts. Each execution separately supplies a canonical launch instance with exact FD values; no caller, inherited default, final-environment self-reference, or alternative config API."
      },
      "scipy": {
        "closed_keys": [
          "configuration_sha256",
          "version",
          "wheel_sha256"
        ],
        "types": "version is nonempty ASCII; hashes bind exact provider report and exact wheel-index row"
      },
      "threads": {
        "closed_keys": [
          "affinity_sha256",
          "blis",
          "mkl",
          "numexpr",
          "omp",
          "openblas",
          "torch_interop",
          "torch_intraop",
          "veclib"
        ],
        "types": "eight thread counts are exact integer one; affinity_sha256 binds the exact sanitized CPU affinity report"
      },
      "torch": {
        "closed_keys": [
          "build_config_sha256",
          "cpu_capability",
          "cuda_available",
          "git_version",
          "version",
          "wheel_sha256"
        ],
        "types": "cuda_available is boolean but execution remains CPU-only; strings are nonempty ASCII and hashes bind exact build report and wheel-index row"
      }
    },
    "floating_point_control": {
      "backend_state": {
        "closed_keys": [
          "mkldnn_deterministic",
          "mkldnn_enabled",
          "nnpack_enabled",
          "torch_deterministic_algorithms",
          "torch_deterministic_warn_only",
          "torch_matmul_precision"
        ],
        "required_values": {
          "mkldnn_deterministic": true,
          "mkldnn_enabled": false,
          "nnpack_enabled": false,
          "torch_deterministic_algorithms": true,
          "torch_deterministic_warn_only": false,
          "torch_matmul_precision": "highest"
        },
        "set_and_readback": "Set through the documented Torch process APIs immediately after import and read every value back immediately before and immediately after each 56-view checkpoint replay. An absent setter/getter, unsupported state, warning-only fallback, changed readback, or use before locking fails."
      },
      "fe_rounding": {
        "required_symbol": "FE_TONEAREST",
        "set_and_readback": "Call fesetround(FE_TONEAREST), require success, then require fegetround()==FE_TONEAREST before the pre-probes, after their MXCSR reset, and after the numerical region. The allowlisted probe binds the platform constant and libc origin; assumed or unreadable state fails."
      },
      "helper": {
        "abi": [
          "fe_tonearest() -> exact platform C int constant for FE_TONEAREST",
          "get_round() -> C int; set_round(C int) -> exact C int zero on success",
          "get_mxcsr() -> uint32; set_mxcsr(uint32) -> None",
          "wrong arity/type/range, exception, absent export, origin mismatch, or extra state-changing export used by replay fails"
        ],
        "build_rule": "The extension is built before P06, never generated at runtime, and its exact binary is environment evidence. No inline machine code, ctypes callback shellcode, JIT, or compiler invocation during replay is allowed.",
        "closed_exports": [
          "fe_tonearest",
          "get_mxcsr",
          "get_round",
          "set_mxcsr",
          "set_round"
        ],
        "module": "pams.temporac._fp_control",
        "origin": "one prebuilt native extension member whose complete bytes/hash appear in the populated wheel and ELF/native indexes and whose PY_EXTENSION_LOAD event is present"
      },
      "mxcsr": {
        "applicability": "required x86_64 target only",
        "exact_hex": "0x00001f80",
        "semantics": "exception flags clear; all exception masks set; round-to-nearest-even; DAZ bit 6 zero; FTZ bit 15 zero",
        "set_and_readback": "Before probes write exact 0x00001f80 and read exact equality. After the pre-probes, rewrite and reread exact 0x00001f80 before the first forward. After the numerical region require (readback & 0xffffffc0)==0x00001f80; sticky exception-status bits 0..5 may reflect arithmetic and are not evidence of control drift. Before any next region clear them by another exact write/read. Generated helper bytes, partial control masks, or inability to read/write fails."
      },
      "probe": {
        "execution": "Construct inputs from the exact uint32 bit patterns, reinterpret as CPU float32 without decimal conversion, execute one Torch CPU scalar multiplication per case after all locks, materialize C-order <f4 output, reinterpret one uint32, and compare exact bits.",
        "failure": "Any exception, unsupported dtype/op, mismatched output bit, or backend/readback drift fails closed before evidence can be PASS.",
        "input_denormal_case": {
          "a_bits": "0x00400000",
          "b_bits": "0x40000000",
          "expected_bits": "0x00800000"
        },
        "order": "output_denormal_case then input_denormal_case immediately before the numerical region, followed by exact MXCSR reset/read; repeat the two probes after the region after control-bit/FE/backend readback",
        "output_denormal_case": {
          "a_bits": "0x00800000",
          "b_bits": "0x3f000000",
          "expected_bits": "0x00400000"
        }
      },
      "torch_flush_denormal": {
        "required_value": false,
        "set_and_verify": "torch.set_flush_denormal(False) must report supported/successful state; exact denormal probes are the mandatory functional readback. Missing API, false success, or probe mismatch fails."
      }
    },
    "frozen_execution": [
      "CPU-only float32 model parameters/operations; binary64 scoring follows the exact named operation DAG",
      "one fresh subprocess for each authority-bearing 56-view checkpoint tune evaluation; one checkpoint per subprocess, no process reuse across checkpoints",
      "the consumed K7 evaluator is itself one fresh non-resumable process and does not resume or delegate vault access; it re-establishes and probes the same controls before each checkpoint replay region",
      "FE_TONEAREST, exact x86 MXCSR 0x00001f80, torch flush-denormal false, denormal bit probes, and deterministic CPU backend state are set/read/probed fail-closed",
      "torch.set_num_threads(1) and torch.set_num_interop_threads(1)",
      "OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=VECLIB_MAXIMUM_THREADS=BLIS_NUM_THREADS=1",
      "OMP_DYNAMIC=FALSE and MKL_DYNAMIC=FALSE",
      "TempoRACTeacher.eval() and torch.inference_mode()",
      "each of 56 views is one unbatched, unpadded CPU float32 forward in view order; no vectorization across views or runs",
      "no autocast, GradScaler, bfloat16, GPU, CUDA kernel, TF32, FMA/reassociation of the specified binary64 score DAG, or approximate reciprocal",
      "torch deterministic algorithms true/warn_only false, matmul precision highest, MKLDNN disabled/deterministic, NNPACK disabled, and all recorded flags exact"
    ],
    "installed_distribution_member_index": {
      "association_preimage": {
        "closed_keys": [
          "bytes",
          "distribution",
          "generated_member_row_sha256_or_null",
          "installed_path",
          "kind",
          "member_tag",
          "normalized_member_path_or_null",
          "repository_path_or_null",
          "repository_sha256_or_null",
          "repository_tag",
          "schema",
          "sha256",
          "version",
          "wheel_member_path_or_null",
          "wheel_member_sha256_or_null",
          "wheel_sha256"
        ],
        "digest_rule": "association_sha256 is SHA-256 of the standalone exact 16-key temporac.installed-member-association.v4 object in closed_keys order-independent canonical compact sorted-key UTF-8 JSON plus one LF. The index row repeats all 15 non-schema values with identical names and adds only association_sha256. Parse/rebuild/re-encode byte equality is mandatory; tagged-H, concatenation, key rename, omitted null, pretty JSON, CRLF, or alternate encoding fails.",
        "schema": "temporac.installed-member-association.v4"
      },
      "completeness": [
        "P06 starts from an empty wheel-installation namespace and exact wheel archives. Decode each non-directory ZIP member by the sole normalization/UTF-8-flag rule, then apply exact PEP-427 purelib/platlib/scripts/headers/data mapping byte-for-byte. Collision, overwrite, symlink, hardlink/case alias, editable install, .egg-link, executable .pth, pyc, __pycache__, installer synthesis, rewrite, or network input fails.",
        "Every row has member_tag=wheel-member and every non-directory wheel member maps exactly once. The installer-generated-member index is present but has exactly zero rows; no member_tag=installer-generated row is accepted.",
        "The enumerated namespace is exactly the paths produced by wheel mapping beneath site-packages and wheel-owned data/bin/include locations. Prebuilt CPython, supervisor, loader and native runtime files are disjoint and proved by CPython/ELF/launcher indexes, not falsely treated as wheel members.",
        "For every row normalized_member_path_or_null equals normalize_rel_posix_ascii(wheel_member_path_or_null), wheel_member_sha256_or_null hashes exact uncompressed member bytes, installed_path equals deterministic PEP-427 mapping, and installed bytes/hash equal the member. repository_tag=repository-member requires exact project-prefix source equality; no-repository-member requires both repository fields null."
      ],
      "installer_generated_member_index": {
        "derivation_tokens": [],
        "expected_row_count": 0,
        "row_closed_keys": [
          "bytes",
          "derivation",
          "distribution",
          "input_member_paths",
          "input_member_sha256",
          "installed_path",
          "output_sha256",
          "schema",
          "version"
        ],
        "rows_required": [],
        "rule": "The exact accepted row array is []. This runtime performs a byte-preserving wheel-member mapping only: it does not synthesize console scripts, rewrite RECORD, emit INSTALLER/direct_url metadata, compile pyc, or invoke an installer. Wheel entry_points metadata remains an ordinary wheel member and creates no bin wrapper. Therefore member_tag=installer-generated and every null-wheel association are structurally defined for rejection but have accepted multiplicity zero.",
        "schema": "temporac.installer-generated-member-index.v4",
        "top_level_closed_keys": [
          "contract_sha256",
          "rows",
          "schema",
          "wheel_index_sha256"
        ]
      },
      "kind_tokens": [
        "data",
        "dist-info",
        "extension",
        "native-library",
        "pure-python",
        "resource",
        "script"
      ],
      "member_tag_tokens": [
        "installer-generated",
        "wheel-member"
      ],
      "normalization": "normalize_rel_posix_ascii(s): require JSON string; strict ASCII bytes 0x21..0x7e; NFC(s)==s; reject NUL, backslash, colon, leading/trailing slash, repeated slash, empty/'.'/'..' segment; split only on '/' and rejoin unchanged. No percent decode, case fold, separator replacement, Unicode-changing normalization or realpath. A ZIP name with bit 0x800 uses strict UTF-8 then must be ASCII; without bit 0x800 every raw name byte must already be ASCII 0x21..0x7e and decodes one-to-one. All other encodings fail. distribution is the already-normalized PEP-503 lowercase ASCII result; version is exact nonempty ASCII METADATA Version.",
      "order": "ASCII distribution, ASCII version, member_tag with installer-generated before wheel-member, null-safe key (generated installed_path for installer-generated; normalized_member_path_or_null for wheel-member), then installed_path",
      "project_distribution": {
        "closed_keys": [
          "distribution",
          "installed_prefix",
          "repository_prefix",
          "version"
        ],
        "distribution": "pams-rac-reproduction",
        "installed_prefix": "site-packages/pams",
        "repository_prefix": "src/pams",
        "version": "0.1.0"
      },
      "project_repository_equality": [
        "Every installed regular project payload member beneath site-packages/pams has exactly one repository row obtained by replacing the installed_prefix with repository_prefix. The installed bytes, positive byte count, and SHA-256 equal that repository row byte-for-byte. Every repository row with purpose runtime or installation-equality beneath src/pams has exactly one installed project row. Project .dist-info and generated console scripts are wheel/installer metadata, not project source, and cannot be imported or executed by the authority entrypoint.",
        "The exact runtime projection is the installed project pure-python rows whose repository row purpose is runtime. It is a bijection with every source-allowlist runtime row beneath src/pams, including src/pams/__init__.py, src/pams/types.py, src/pams/temporac/__init__.py, every reachable TempoRAC module, and future protocol_entry.py. An installation-equality or verifier-test row is not an executable authorization. Any trace event for it fails.",
        "Noneditable installation is mandatory. No current working tree, cwd, PYTHONPATH, user site, .pth, namespace overlay, import hook, zip import, or alternate pams distribution may contribute an origin."
      ],
      "repository_tag_tokens": [
        "no-repository-member",
        "repository-member"
      ],
      "row_closed_keys": [
        "association_sha256",
        "bytes",
        "distribution",
        "generated_member_row_sha256_or_null",
        "installed_path",
        "kind",
        "member_tag",
        "normalized_member_path_or_null",
        "repository_path_or_null",
        "repository_sha256_or_null",
        "repository_tag",
        "sha256",
        "version",
        "wheel_member_path_or_null",
        "wheel_member_sha256_or_null",
        "wheel_sha256"
      ],
      "schema": "temporac.installed-distribution-member-index.v4",
      "top_level_closed_keys": [
        "contract_sha256",
        "installer_generated_member_index_sha256",
        "project_distribution",
        "rows",
        "schema",
        "source_allowlist_sha256",
        "wheel_index_sha256"
      ],
      "validation": [
        "member_tag and repository_tag are mandatory exact discriminants; nullable fields have no inferred third state.",
        "Every accepted row has member_tag=wheel-member, generated_member_row_sha256_or_null=null, and three nonnull wheel fields whose normalized/raw relation is exact. installer-generated has multiplicity zero and any instance fails.",
        "repository-member requires both repository fields nonnull and exact source-allowlist byte/hash equality; no-repository-member requires both null.",
        "Every reconstructed association preimage parses/re-encodes byte-identically; association digests are duplicate-free; project mappings, wheel members, installed rows and trace references are bijections."
      ]
    },
    "platform_rule": "The replay platform must report platform_system exact Linux, architecture.machine exact x86_64, architecture.byteorder exact little, and pointer_bits exact 64. Any other value fails before artifact construction.",
    "population_rule": "Future P3/P06 must populate every actual version, wheel archive, installed member, CPython stdlib/core member, trusted generated source, supervisor/ELF, build, CPU, dispatch, helper, and environment value from a fresh preflight and serialize the exact artifacts. This amendment supplies requirements and literal launch layout only; it fabricates no digest or measured runtime value. Missing, placeholder, null where not expressly allowed, inferred, unsupported, unreadable, non-bijective, or untraced values fail.",
    "required_bindings": [
      "exact native supervisor path/argv/bytes and every full/discovery launch-instance FD state",
      "exact CPython 3.12.13 PyPreConfig/PyConfig constructors, complete field maps, setters, call order, native/runtime readbacks, error handling, clear/finalize",
      "exact Torch, NumPy, SciPy, and every imported distribution wheel archive plus complete typed installed-distribution member association bytes",
      "complete CPython standard-library source/extension inventory and executed builtin/frozen provider inventory",
      "exact project installed-member to repository allowlist byte equality, including pams/__init__.py and pams/types.py before protocol_entry",
      "acyclic bootstrap-only dataclasses discovery capsule/precommit/exact two-row launch-readback-output-attestation evidence index/root, two equal nonempty outputs/root, and exact trusted generated rows",
      "complete ELF/shared-library index for supervisor, interpreter/libpython, Python native extensions, BLAS, OpenMP, oneDNN, libc, libm, libstdc++, dynamic loader, and transitive loaded objects",
      "BLAS vendor/build/config and oneDNN version/build/primitive ISA",
      "CPU architecture, model, microcode-visible flags, ISA feature list, Torch CPU capability and dispatch tables",
      "thread counts and affinity plus environment-variable values",
      "deterministic-algorithm flags, matmul precision, autocast state, CUDA availability/use, and TF32 flags",
      "active floating-point control: libc rounding mode, x86 MXCSR, Torch denormal behavior, exact denormal probes, and CPU backend enable/determinism switches"
    ],
    "runtime_launcher": {
      "argv": [
        "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
        "--embedded-cpython-v4"
      ],
      "closed_keys": [
        "argv",
        "cpython_initialization_profile_sha256",
        "envp",
        "executable_bytes",
        "executable_path",
        "executable_sha256",
        "fd_contract_sha256",
        "launch_instance_schema",
        "module_search_paths",
        "preimport_sequence",
        "profile_name",
        "schema",
        "working_directory"
      ],
      "envp": [
        "BLIS_NUM_THREADS=1",
        "LANG=C",
        "LC_ALL=C",
        "MKL_DYNAMIC=FALSE",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "OMP_DYNAMIC=FALSE",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "PYTHONHASHSEED=0",
        "TZ=UTC",
        "VECLIB_MAXIMUM_THREADS=1"
      ],
      "executable_path": "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
      "launch_instance_schema": "temporac.native-launch-instance.v4",
      "module_search_paths": [
        "/opt/temporac/replay-v4/lib/python3.12",
        "/opt/temporac/replay-v4/lib/python3.12/lib-dynload",
        "/opt/temporac/replay-v4/lib/python3.12/site-packages"
      ],
      "preimport_sequence": [
        "The controller validates supervisor/ELF, accepted contract, source, environment, initialization/FD profiles and constructs a post-environment launch instance binding actual payload bytes/hashes without feeding back into environment_sha256.",
        "At process entry the supervisor validates /proc/self/exe, AT_EXECFN, initial images, exact descriptor state, envp/argv/cwd; emits PROCESS_IMAGE/PREINIT_NATIVE_IMAGE and exec-entry NATIVE_FD_READBACK; installs PySys_AddAuditHook; locks/validates/closes specified FDs; emits locked-pre-cpython NATIVE_FD_READBACK; then follows the exact PyPreConfig/PyConfig sequence.",
        "Post-initialize complete struct/_Py_GetConfigsAsDict/sys-stream and FD readbacks emit CPYTHON_CONFIG_READBACK and NATIVE_FD_READBACK and must match before PyConfig_Clear and authority imports. Full runtime then consumes the exact nonempty trusted-generated row sequence during the dataclasses bootstrap, imports the remaining exact closure with no additional generated event, emits DISPATCH_BEGIN and dispatches.",
        "Every executable origin resolves through the unique typed canonical association/origin preimage. Trace/launch/readback evidence is post-environment and never authorizes its launcher, discovery capsule, or precommit."
      ],
      "profile_name": "full-runtime",
      "schema": "temporac.native-isolated-launch.v4",
      "working_directory": "/"
    },
    "schema": "temporac.cpu-replay-environment.v4",
    "status": "FUTURE_P3_VALUE_REQUIRED_NOT_PRESENT",
    "top_level_closed_keys": [
      "architecture",
      "autocast",
      "blas",
      "code_index_sha256",
      "contract_sha256",
      "cpu_dispatch",
      "cpu_isa",
      "cpython_initialization_profile_sha256",
      "cpython_runtime_member_index_sha256",
      "determinism",
      "elf_index_sha256",
      "environment_variables",
      "fd_contract_sha256",
      "floating_point_control",
      "generated_source_discovery_capsule_sha256",
      "generated_source_discovery_evidence_index_sha256",
      "generated_source_discovery_evidence_root_sha256",
      "generated_source_discovery_output_root_sha256",
      "generated_source_discovery_precommit_sha256",
      "installed_distribution_member_index_sha256",
      "numpy",
      "onednn",
      "platform_system",
      "python",
      "runtime_launcher",
      "schema",
      "scipy",
      "threads",
      "torch",
      "trusted_generated_source_index_sha256",
      "wheel_index_sha256"
    ],
    "trusted_generated_source_index": {
      "bootstrap_profile": "bootstrap-only-dataclasses",
      "compile_mode_tokens": [
        "eval",
        "exec"
      ],
      "completeness": "Rows are exactly and byte-for-byte the common nonempty row array of both independently evidenced bootstrap-only dataclasses discovery outputs. Every full-runtime process must reproduce and consume this finite ordered stdlib-generated list one row at a time before DISPATCH_BEGIN and observe no other anonymous generated event. Empty trusted set, project/site-packages import in discovery, caller dataclass generation, missing/extra/reordered/different output, row learned from final runtime, or any generated event after DISPATCH_BEGIN fails.",
      "discovery_contract": {
        "capsule": {
          "argv": [
            "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
            "--discover-generated-v4"
          ],
          "closed_keys": [
            "argv",
            "bootstrap_modules",
            "contract_sha256",
            "cpython_initialization_profile_sha256",
            "cpython_runtime_member_index_sha256",
            "dataclasses_row",
            "envp",
            "fd_contract_sha256",
            "initial_native_image_index_sha256",
            "interpreter_image_sha256",
            "libpython_image_sha256",
            "module_search_paths",
            "schema",
            "supervisor_bytes",
            "supervisor_path",
            "supervisor_sha256",
            "working_directory"
          ],
          "dataclasses_row_closed_keys": [
            "bytes",
            "kind",
            "normalized_path",
            "provider_path",
            "sha256"
          ],
          "dataclasses_row_required": {
            "bytes": 62085,
            "kind": "stdlib-source",
            "normalized_path": "stdlib/dataclasses.py",
            "provider_path": "lib/python3.12/dataclasses.py",
            "sha256": "d242aea5fcf6408b1c1f622442f88f68b9526ce1f8bd2890d74a144677c427d9"
          },
          "dependency_rule": "The capsule is constructed only from accepted effective-contract bytes, the exact supervisor image, exact CPython 3.12.13 interpreter/libpython/stdlib/dataclasses rows, the already closed initialization/FD profiles, and an initial-native-image index. It contains no source_allowlist_sha256, installed project/wheel index, trusted-generated index, final environment, final code index, trace, G5a, receipt, data, or artifact digest.",
          "digest_rule": "capsule_sha256 is SHA-256 of the exact canonical capsule bytes including LF. Every nested object has the closed keys below and is copied from its already validated pre-environment ledger; no digest-only substitute for dataclasses or supervisor bytes is allowed.",
          "initial_native_image_index": {
            "row_closed_keys": [
              "build_id_or_null",
              "bytes",
              "path",
              "sha256",
              "soname"
            ],
            "rule": "Rows are the exact ASCII absolute-path ordered supervisor exec-entry dl_iterate_phdr projection, including supervisor, dynamic loader, libpython and every transitive initially mapped ELF; every row byte-equals the already built CPU ELF row. No project/site-packages image or final environment/code digest participates.",
            "schema": "temporac.initial-native-image-index.v4",
            "top_level_closed_keys": [
              "contract_sha256",
              "rows",
              "schema"
            ]
          },
          "literal_values": {
            "argv": [
              "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
              "--discover-generated-v4"
            ],
            "bootstrap_modules": [
              "dataclasses"
            ],
            "envp": [
              "LANG=C",
              "LC_ALL=C",
              "PYTHONHASHSEED=0",
              "TZ=UTC"
            ],
            "module_search_paths": [
              "/opt/temporac/replay-v4/lib/python3.12",
              "/opt/temporac/replay-v4/lib/python3.12/lib-dynload"
            ],
            "working_directory": "/"
          },
          "schema": "temporac.generated-source-discovery-capsule.v4",
          "supervisor_rule": "supervisor_path is the exact absolute regular non-symlink path in argv[0]; supervisor_bytes is positive and supervisor_sha256 hashes its complete bytes. The same three values equal the process-image and initial-native-image records in both discovery runs."
        },
        "evidence": {
          "aggregate": {
            "digest_rule": "discovery_evidence_index_sha256 is SHA-256 of the complete canonical temporac.generated-source-discovery-evidence-index.v4 bytes including LF. The index contains no own digest or root field.",
            "root_rule": "discovery_evidence_root_sha256=H(temporac.generated-source-discovery-evidence-root.v4, exact complete evidence-index bytes including LF). No row digest list, shared-output digest, launch digest or alternate packing substitutes for the index bytes."
          },
          "attestation": {
            "closed_keys": [
              "capsule_sha256",
              "child_pid",
              "child_starttime_ticks",
              "configuration_readback_sha256",
              "discovery_output_root_sha256",
              "discovery_output_sha256",
              "exec_entry_fd_readback_sha256",
              "exit_code",
              "exit_kind",
              "fd3_eof_observed",
              "fd3_stream_sha256",
              "launch_instance_sha256",
              "locked_pre_cpython_fd_readback_sha256",
              "post_pyinitialize_fd_readback_sha256",
              "precommit_sha256",
              "process_ordinal",
              "run_id",
              "schema",
              "signal_or_null",
              "status",
              "supervisor_sha256",
              "wait_status_u32"
            ],
            "freshness_and_join": "The controller calls the exact absolute supervisor exec twice, sequentially. For run 0 it opens a fresh output pipe, spawns one child, records positive child_pid and child_starttime_ticks, drains/validates the complete FD3 frame stream to EOF, then waitpid(child_pid,0) and commits row 0 before creating any run-1 pipe/process. It repeats for run 1. The two (child_pid,child_starttime_ticks) pairs must differ; no fork-without-exec, same-process reset, shared pipe, pid substitution, overlapping writer or reused read peer is accepted.",
            "required_values": "run_id and process_ordinal are the same exact integer 0 or 1; exit_kind is exited; exit_code=0; signal_or_null=null; wait_status_u32=0; fd3_eof_observed=true; status=JOINED_PASS. Every named digest equals the same-named evidence-row digest; the three FD stages are committed separately and no aggregate/digest list substitutes for them. EOF before five complete frames, bytes after frame five, wrong wait target/status, signal, nonzero exit, missing Py_FinalizeEx-success path, or unattested launch fails.",
            "schema": "temporac.generated-source-discovery-run-attestation.v4"
          },
          "fd3_framing": {
            "caps": "Each canonical JSON payload is positive and at most 16 MiB; the complete stream is at most 64 MiB. Length is checked before allocation. A zero/oversize/truncated frame fails and grants no output authority.",
            "frame_encoding": "The stream is ASCII temporac.generated-source-discovery.fd3.v4, then one 0x00, uint16_be(5), then exactly five frames. Each frame is uint8 type_code || uint64_be(payload_length) || raw32(SHA256(payload_bytes)) || payload_bytes. Payload bytes are the complete strict canonical JSON object including exactly one LF. The reader consumes exact lengths concurrently and rejects short read, extra byte, duplicate/type reorder, wrong digest/schema, alternate endian or concatenated JSON.",
            "frame_order": [
              {
                "payload_schema": "temporac.linux-fd-readback.v4",
                "required_stage": "exec-entry",
                "type_code": 1
              },
              {
                "payload_schema": "temporac.linux-fd-readback.v4",
                "required_stage": "locked-pre-cpython",
                "type_code": 2
              },
              {
                "payload_schema": "temporac.linux-fd-readback.v4",
                "required_stage": "post-pyinitialize",
                "type_code": 3
              },
              {
                "payload_schema": "temporac.cpython-initialization-readback.v4",
                "required_stage": "post-pyinitialize",
                "type_code": 4
              },
              {
                "payload_schema": "temporac.generated-source-discovery-output.v4",
                "required_stage": "complete-output",
                "type_code": 5
              }
            ],
            "independent_streams": "Run 0 and run 1 each have their own newly created controller read peer and child fd3 write end. No frame from one process may occur in, prefix, suffix or authorize the other stream. Child closes fd3 only after frame five; controller requires terminal EOF and joined exit before canonical attestation.",
            "sha_rule": "fd3_stream_bytes is the exact positive byte count of the reconstructed complete stream and fd3_stream_sha256 hashes every stream byte. The five *_bytes_hex evidence fields decode to the exact payloads and must reconstruct this one stream byte-for-byte."
          },
          "index": {
            "equality_rules": [
              "Both rows have identical capsule_sha256, precommit_sha256 and supervisor_sha256 and resolve the same precommitted bytes.",
              "Their complete configuration-readback bytes/hashes are byte-identical. Their launch instances each validate the same semantic mode/argv/cwd/env/final FD rows; construction-index transient fd numbers, launch hashes, run/process identity and attestations are instance-specific and may differ.",
              "For each of the three FD stages, after replacing only launch_instance_sha256 by a fixed zero digest, canonical readback bytes are byte-identical between runs; the unmodified bytes/hash in each row must name its own exact launch instance. No other difference is allowed.",
              "discovery_output_bytes_hex, discovery_output_sha256 and discovery_output_root_sha256 are each byte-identical between rows and equal the top-level shared values. rows is nonempty and every row is byte-identical in ordinal order.",
              "Attestation fields other than run_id, process_ordinal, child_pid, child_starttime_ticks, launch_instance_sha256, fd3_stream_sha256 and the three readback-dependent digests are byte-identical. The two fresh-process identities and run ordinals must differ exactly as specified."
            ],
            "row_closed_keys": [
              "attestation_bytes_hex",
              "attestation_sha256",
              "capsule_sha256",
              "configuration_readback_bytes_hex",
              "configuration_readback_sha256",
              "discovery_output_bytes_hex",
              "discovery_output_root_sha256",
              "discovery_output_sha256",
              "exec_entry_fd_readback_bytes_hex",
              "exec_entry_fd_readback_sha256",
              "fd3_stream_bytes",
              "fd3_stream_sha256",
              "launch_instance_bytes_hex",
              "launch_instance_sha256",
              "locked_pre_cpython_fd_readback_bytes_hex",
              "locked_pre_cpython_fd_readback_sha256",
              "post_pyinitialize_fd_readback_bytes_hex",
              "post_pyinitialize_fd_readback_sha256",
              "precommit_sha256",
              "process_ordinal",
              "run_id",
              "schema",
              "supervisor_sha256"
            ],
            "row_rule": "Every *_bytes_hex value is lowercase even-length hex decoding to the complete exact canonical bytes including LF, except fd3_stream_bytes which is the exact positive integer byte count. Every paired digest/root and embedded capsule/precommit/launch/stage/schema/run value is independently recomputed. Row schema is exact temporac.generated-source-discovery-evidence-row.v4.",
            "rows": "Exactly two rows ordered run_id 0 then 1; process_ordinal equals run_id. Missing, extra, duplicate, reordered, reused-process, reused-pipe, digest-only, noncanonical, unjoined, output-only or cross-row-substituted evidence fails.",
            "schema": "temporac.generated-source-discovery-evidence-index.v4",
            "top_level_closed_keys": [
              "capsule_sha256",
              "contract_sha256",
              "precommit_sha256",
              "rows",
              "schema",
              "shared_output_root_sha256",
              "shared_output_sha256"
            ]
          }
        },
        "output": {
          "exact_bytes_rule": "Each discovery serializes exactly capsule_sha256, the complete rows sequence, and schema as canonical compact sorted-key UTF-8 JSON plus one LF. The controller recomputes output_sha256 and the tagged output root; the two fresh outputs must be byte-identical. No placeholder/guessed output digest is precommitted.",
          "expected_rows": "rows is the complete finite nonempty ordered sequence of generated-source units caused solely by importing exact dataclasses and its CPython-stdlib transitive imports under the capsule. One row is created at each actual non-file-backed source compile in compile-event order and is retained only when that code object is immediately consumed by exactly one matching eval/exec event; ordinal is zero-based and gap-free. Exact CPython 3.12.13 dataclasses->inspect->collections.namedtuple creates at least one such unit. The sequence cannot be empty, predicted, caller-supplied, truncated, filtered by qualname, coalesced across distinct compiles, duplicated, or expanded by any other action.",
          "root_rule": "output_sha256 hashes complete canonical output bytes including LF. output_root_sha256=H(temporac.generated-source-discovery-root.v4, raw32(capsule_sha256), exact output bytes). Both fresh discoveries must produce byte-identical output bytes and the same root.",
          "row_closed_keys": [
            "compile_filename",
            "compile_mode",
            "generated_source_hex",
            "generated_source_sha256",
            "generator_origin_class",
            "generator_qualname",
            "generator_sha256",
            "kind",
            "ordinal",
            "profile",
            "source_bytes"
          ],
          "schema": "temporac.generated-source-discovery-output.v4",
          "top_level_closed_keys": [
            "capsule_sha256",
            "rows",
            "schema"
          ]
        },
        "precommit": {
          "closed_keys": [
            "capsule_bytes",
            "capsule_sha256",
            "contract_sha256",
            "created_before_process_count",
            "schema"
          ],
          "required_value": "created_before_process_count is exact integer zero. The immutable precommit bytes/hash, including the complete capsule bytes/hash and contract digest, are durably committed before either discovery exec; neither output nor final-environment/code/trusted-index digest exists or appears in the precommit.",
          "schema": "temporac.generated-source-discovery-precommit.v4"
        },
        "procedure": [
          "P06 builds and validates the exact initialization/FD profiles, CPython runtime/dataclasses rows, initial-native-image index, supervisor image and capsule, then durably writes the exact precommit with process count zero before either discovery exec.",
          "Launch run 0 and then run 1 as two sequential fresh exec processes with distinct pipes/process identities, exact --discover-generated-v4 argv, four-FD discovery layout, literal envp/cwd and discovery PyConfig. The native hook precedes Py_PreInitialize. Import exactly dataclasses as the sole requested module and allow only its CPython-stdlib transitive imports; project/site-packages import, dataclass application to a caller class, final environment/code/trusted-index access, dispatch, data, receipt or artifact access fails.",
          "Each child emits exactly the five length-prefixed FD3 frames. The controller reconstructs every frame, validates exact launch/config/all-three-FD readbacks, complete nonempty generated output, terminal EOF and joined exit, then constructs the exact attestation and one evidence row. Row 0 is committed before run 1 starts. No shared stream, concatenated JSON, process reuse or output-only proof is accepted.",
          "Canonicalize the exact two-row evidence index, require all equality/difference rules, compute its digest and tagged root, and only then derive trusted rows from the byte-identical nonempty shared output. The trusted index and final CPU environment bind both evidence-index digest and root.",
          "The future full-runtime source closure may reproduce only these exact stdlib-generated rows before DISPATCH_BEGIN and no anonymous generated row afterwards. Current source is non-authorizing and may require separately reviewed refactoring that predefines project dataclass-generated methods; silently learning rows from project/final imports is forbidden.",
          "Only after capsule/precommit/two outputs/trusted index are fixed may the final CPU environment be serialized. Execute a distinct full P06 invocation under --embedded-cpython-v4; validate final source/environment, consume every trusted row exactly once in order, follow ordinary dispatch/receipt, and bind launch/readback/trace evidence. Discovery output never authorizes a process that produced it."
        ],
        "schema": "temporac.generated-source-discovery-contract.v4"
      },
      "full_runtime_order": [
        "Immediately after initialization/readbacks and before importing pams, import exact dataclasses once under profile bootstrap-only-dataclasses. For trusted row ordinal i, the next anonymous-source action must emit PY_TRUSTED_GENERATED_COMPILE with the exact row-derived origin, then exactly one immediately matching PY_TRUSTED_GENERATED_EXEC with the byte-identical origin; only after that pair is row i consumed and i+1 may begin.",
        "The consumed row ordinal sequence is exactly 0..R-1 for positive R equal to both discovery outputs. Each row is consumed once and only once; no generic PY_CODE_EVAL/PY_CODE_EXEC trusted-generated alternative, lookahead, retry, duplicate, omission, reordering, source coalescing or unmatched compile/exec is allowed.",
        "Then import pams, pams.types, pams.temporac, protocol_entry, and the remaining ASCII source-allowlist runtime module order. dataclasses is already cached; any project/third-party anonymous generation during these imports fails.",
        "After exact closure validation emit DISPATCH_BEGIN and dispatch. No generated event is accepted after the trusted bootstrap sequence."
      ],
      "generated_source_encoding": "The audit hook receives str or bytes. str is encoded once as strict UTF-8 with no BOM; bytes are used verbatim. generated_source_hex is lowercase hex of those exact bytes, source_bytes is its positive length, and generated_source_sha256 hashes them. NUL, undecodable filename, code-object-only execution, marshal/pickle code, or source over 1 MiB fails.",
      "generator_origin_class_tokens": [
        "cpython-stdlib-source"
      ],
      "generator_origin_mapping": {
        "class_to_kind": {
          "cpython-stdlib-source": "cpython-stdlib"
        },
        "digest_equality": "For the sole mapping, trusted-row generator_sha256 equals origin.generator_origin_sha256 byte-for-byte and equals runtime_member_sha256 of exactly one cpython-runtime-member-index stdlib-source row. Renaming, rehashing an origin object, provider digest substitution, prefix/suffix packing or any other digest projection is forbidden.",
        "origin_membership": "origin.generator_origin_kind is exactly cpython-stdlib. Its generator_origin_sha256 plus generator_qualname and compile_filename resolve one already traced CPython stdlib source generator; project-installed, installed-wheel, native, process, generated or digest-only generator origin fails. The mapping is one-to-one and exhaustive."
      },
      "kind_token": "trusted-bootstrap-generated-source",
      "mandatory_runtime_classification": "Only exact CPython-3.12.13 stdlib generation caused by the capsule's dataclasses import is admitted. The trusted set is required nonempty and equals both discovery row arrays exactly. Generator source/hash/qualname, compile filename/mode, generated bytes, profile, ordinal and standalone trusted-row digest all match one row; the row is consumed by its one required adjacent trusted compile-to-exec pair. Project/third-party generator identity and unseen output are never authorized.",
      "noncircularity": "Initialization/FD/CPython/native inputs precede the capsule and its zero-process precommit; two fresh discoveries produce the exact two-row evidence index and equal nonempty output; that evidence/output produces this trusted index; only then may the final code/environment and full supervisor run. The capsule has no evidence/final environment/code/trusted-index/trace digest, and the final environment refers forward to capsule/precommit/evidence/output/trusted digests. No discovery result authorizes either process that produced it.",
      "order": "ASCII profile then numeric ordinal",
      "row_closed_keys": [
        "compile_filename",
        "compile_mode",
        "generated_source_hex",
        "generated_source_sha256",
        "generator_origin_class",
        "generator_qualname",
        "generator_sha256",
        "kind",
        "ordinal",
        "profile",
        "source_bytes"
      ],
      "row_hash_preimage": {
        "digest": "trusted_row_sha256=SHA-256 over every byte of the exact standalone row object, including its one terminal LF.",
        "exact_bytes": "Construct one standalone object containing exactly the eleven keys in row_closed_keys with the exact values from the committed trusted-index row, including generated_source_hex. Serialize by the amendment canonical compact recursively sorted-key UTF-8 JSON rule and append exactly one LF. The row has no schema key and no embedded trusted_row_sha256.",
        "forbidden": "No parent/index bytes, discovery-evidence row, output-root tag, ordinal prefix, H framing, pretty JSON, omitted generated_source_hex, renamed key, CRLF, LF-excluded digest or digest-only row substitute is permitted.",
        "index_equality": "The trusted row, discovery run-0 row and discovery run-1 row parse to exactly the same eleven-key value tree and each recanonicalizes to the same standalone bytes. trusted_row_sha256 is unique within the nonempty ordered index; a collision or alternate object with the same digest fails.",
        "name": "trusted_row_sha256"
      },
      "schema": "temporac.trusted-generated-source-index.v4",
      "top_level_closed_keys": [
        "contract_sha256",
        "cpython_runtime_member_index_sha256",
        "discovery_capsule_bytes",
        "discovery_capsule_sha256",
        "discovery_evidence_index_bytes",
        "discovery_evidence_index_sha256",
        "discovery_evidence_root_sha256",
        "discovery_output_bytes",
        "discovery_output_root_sha256",
        "discovery_output_sha256",
        "discovery_precommit_sha256",
        "rows",
        "schema"
      ]
    },
    "wheel_index": {
      "closed_keys": [
        "contract_sha256",
        "rows",
        "schema"
      ],
      "order": "ASCII distribution then filename",
      "row_closed_keys": [
        "bytes",
        "distribution",
        "filename",
        "sha256",
        "version"
      ],
      "schema": "temporac.cpu-wheel-index.v4"
    }
  },
  "disposition": {
    "final_status": "PROPOSED_PENDING_ROUND7_REVIEW",
    "implementation_required_after_acceptance": "Only after an independent Round7 ACCEPT may a separately authorized implementation/review create the successor effective-contract index, native build/install outputs, BASE/FINAL bundles, controller/child, FD6/7/8 transport, discovery evidence, dispatch ABI, audit trace, artifacts, receipts, indexes, tests, and replay evidence. Current source/tests and accepted Wave1 postfix remain bound non-authorizing inputs.",
    "no_actions_taken": [
      "no source-code or test-file change and no test execution",
      "no plan, tracker, proposal, review, MANIFEST, trace, or Git change",
      "no server, data, sealed, heldout, vault, result, checkpoint, prediction, target, evaluator, training, or experiment access",
      "no authority, launch, gate, claim, result, job, threshold, data, or method expansion"
    ],
    "review_required": "A fresh independent Round7 specification review must inspect the exact timestamped MD/JSON pair and verify R7-B1 through R7-B9 while confirming the preserved 10+64 CPython profile, floating-point controls, typed association preimages, 54-owner/106-edge projection and every already closed scientific/interface semantic remain unchanged before any implementation, acceptance-derived authority, or execution."
  },
  "effective_contract": {
    "active_accepted": {
      "bytes": 665,
      "role_rows": [
        {
          "bytes": 79366,
          "role": "canonical_proposal",
          "sha256": "3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391"
        },
        {
          "bytes": 312728,
          "role": "amendment_markdown",
          "sha256": "2eaa3cfb746804fdc5890fec4c252c877a031a4abedef7b9ef6eeac0a4839395"
        },
        {
          "bytes": 261196,
          "role": "amendment_json",
          "sha256": "638efc8708f74c8514247f4f4c5e2311595e90edc6744c9b371886f0789af164"
        },
        {
          "bytes": 15666,
          "role": "accepted_review_markdown",
          "sha256": "4a3199c3ac18e3fd2a5bb7ecf6d66e10bc4a5cebb82363803c47867234b70e67"
        },
        {
          "bytes": 10036,
          "role": "accepted_review_json",
          "sha256": "fcf8a9d4949f6b456f18bfb4591c512b6628c6ec7eb5a055873a995992f46ad6"
        }
      ],
      "schema": "temporac.effective-contract-index.v4",
      "sha256": "c203f4aab00879acf041b90829f0124d0b1de231294f191b03f3166f728e3d5c"
    },
    "circularity_policy": "No placeholder digest, amendment self-hash, accepted-review self-reference, fixpoint, mutable-alias row, or post-run backfill is permitted.",
    "prospective_round7_successor": {
      "construction": [
        "The successor index does not exist until a fresh independent Round7 review accepts this exact timestamped pair.",
        "After acceptance, each row binds the positive exact byte count and SHA-256 of the complete named bytes in the listed role order.",
        "The successor contract_sha256 is SHA-256 of the complete canonical index bytes including LF; the index has no contract_sha256 member and none of its five constituents may contain the resulting digest.",
        "All newly built Round7 native provenance artifacts bind only the accepted successor. Existing historical science/operator/X0 artifacts retain their already accepted active digest and are never silently relabeled.",
        "Fixed aliases, the Round6 archive, rejected reviews, Wave1 implementation reviews, and traces are bindings or history but are not successor rows."
      ],
      "current_value": "UNASSIGNED_UNTIL_FRESH_ROUND7_ACCEPT",
      "role_order": [
        {
          "path": "refine-logs/temporac/FINAL_PROPOSAL.md",
          "role": "canonical_proposal"
        },
        {
          "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_215501.md",
          "role": "amendment_markdown"
        },
        {
          "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_215501.json",
          "role": "amendment_json"
        },
        {
          "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND7.md",
          "role": "accepted_review_markdown"
        },
        {
          "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND7.json",
          "role": "accepted_review_json"
        }
      ]
    },
    "row_closed_keys": [
      "bytes",
      "role",
      "sha256"
    ],
    "schema": "temporac.effective-contract-index.v4"
  },
  "executed_code_closure": {
    "completion_index": {
      "byte_fields": "trace_bytes is the exact positive JSON integer byte count of separately supplied canonical trace bytes including LF; trace_sha256 hashes those complete bytes.",
      "order": "ASCII entrypoint_owner",
      "post_g5a": "G5A-COMMIT, G5B-JOIN, and K7-FINAL traces are finalized only after their calls return and therefore cannot be ancestors of their own receipts. Online membership remains mandatory; these three traces are surrounding audit evidence only and never retroactively alter source/runtime indexes, environment, G5a root, capability, result, or authority.",
      "pre_g5a": "The G5a-bound index contains all completed P00 through NATP and all 27 run entrypoints. Each row binds complete launch-instance and trace bytes/hashes; FD/config/origin membership validates against precommitted contract/source/environment/CPython/discovery/installed/wheel/ELF ledgers.",
      "row_closed_keys": [
        "entrypoint_owner",
        "launch_instance_bytes",
        "launch_instance_sha256",
        "trace_bytes",
        "trace_sha256"
      ],
      "schema": "temporac.executable-origin-trace-completion-index.v4",
      "top_level_closed_keys": [
        "contract_sha256",
        "environment_sha256",
        "rows",
        "schema",
        "source_allowlist_sha256"
      ]
    },
    "executable_origin_trace": {
      "capture_boundary": "The finite boundary begins with prevalidated supervisor and an exact launch-instance digest. Root PROCESS_IMAGE is sequence zero, all initial images and descriptor/config readbacks precede CPython initialization, the native hook precedes Py_PreInitialize, and every later imported/executed/native/generated/process origin is emitted through joined exit. Missing launch instance/start/end/readback/boundary/child join or any origin-preimage mismatch fails.",
      "closed_keys": [
        "contract_sha256",
        "entrypoint_owner",
        "environment_sha256",
        "events",
        "launch_instance_sha256",
        "process_count",
        "schema",
        "source_allowlist_sha256"
      ],
      "event_closed_keys": [
        "event",
        "origin",
        "origin_sha256",
        "process_ordinal",
        "sequence",
        "sha256"
      ],
      "event_order": "numeric process_ordinal then zero-based sequence; pairs are unique/gap-free. PROCESS_IMAGE is sequence zero. Exec-entry NATIVE_FD_READBACK, PREINIT_NATIVE_IMAGE and locked-pre-cpython NATIVE_FD_READBACK precede Py_PreInitialize; post-pyinitialize NATIVE_FD_READBACK and CPYTHON_CONFIG_READBACK precede any Python authority import; parent PROCESS_EXEC precedes child PROCESS_IMAGE.",
      "event_semantics": {
        "CPYTHON_CONFIG_READBACK": "origin kind runtime-readback; stage post-pyinitialize; exact initialization-profile, launch-instance, complete config/global/sys/stream readback bytes/hash",
        "CTYPES_DLOPEN": "origin kind native-image; exact path/build-id/ELF digest; one event immediately before load",
        "DISPATCH_BEGIN": "origin kind dispatch-boundary; owner/source digest exact; occurs after all init/import/generated checks and before input-index dereference",
        "NATIVE_FD_READBACK": "origin kind runtime-readback; one event for each exact stage exec-entry, locked-pre-cpython, post-pyinitialize; complete FD readback bytes/hash and launch instance exact",
        "PREINIT_NATIVE_IMAGE": "origin kind process-image for supervisor or native-image for every initial mapping; exact path/build-id/ELF membership",
        "PROCESS_EXEC": "origin kind process-image for the exact same supervisor and next child launch-instance; no other image",
        "PROCESS_IMAGE": "origin kind process-image; exact /proc/self/exe, AT_EXECFN, launcher and ELF identity",
        "PY_BUILTIN_IMPORT": "origin kind cpython-core with provider_kind builtin-provider",
        "PY_CODE_EVAL": "origin kind matches an already emitted file-backed cpython/project/wheel source; trusted-generated is forbidden here and must use the dedicated adjacent trusted pair; raw code-object-only eval fails",
        "PY_CODE_EXEC": "origin kind matches exact file-backed cpython/project/wheel source; trusted-generated is forbidden here and must use the dedicated adjacent trusted pair; raw or unmatched code object fails",
        "PY_EXTENSION_LOAD": "origin kind cpython-extension or installed-extension and exact ELF/crosswalk membership",
        "PY_FROZEN_IMPORT": "origin kind cpython-core with provider_kind frozen-provider",
        "PY_IMPORT_SOURCE": "origin kind cpython-stdlib, installed-wheel, or project-installed; complete source bytes/hash and crosswalk exact",
        "PY_SOURCE_COMPILE": "same typed origin and digest as the unique preceding source event",
        "PY_TRUSTED_GENERATED_COMPILE": "origin kind trusted-generated; complete discovery/root/generator/source identity reconstructs the next unconsumed trusted row and trusted_row_sha256 exact standalone preimage",
        "PY_TRUSTED_GENERATED_EXEC": "same byte-identical trusted-generated origin as the immediately preceding compile; consumes that row exactly once; row ordinals are gap-free and no generated event occurs after DISPATCH_BEGIN"
      },
      "event_sha256_projection": {
        "CPYTHON_CONFIG_READBACK": {
          "runtime-readback": "readback_sha256"
        },
        "CTYPES_DLOPEN": {
          "native-image": "elf_sha256"
        },
        "DISPATCH_BEGIN": {
          "dispatch-boundary": "source_allowlist_sha256"
        },
        "NATIVE_FD_READBACK": {
          "runtime-readback": "readback_sha256"
        },
        "PREINIT_NATIVE_IMAGE": {
          "native-image": "elf_sha256",
          "process-image": "elf_sha256"
        },
        "PROCESS_EXEC": {
          "process-image": "elf_sha256"
        },
        "PROCESS_IMAGE": {
          "process-image": "elf_sha256"
        },
        "PY_BUILTIN_IMPORT": {
          "cpython-core": "provider_sha256"
        },
        "PY_CODE_EVAL": {
          "cpython-stdlib": "runtime_member_sha256",
          "installed-wheel": "installed_member_sha256",
          "project-installed": "installed_member_sha256"
        },
        "PY_CODE_EXEC": {
          "cpython-core": "provider_sha256",
          "cpython-stdlib": "runtime_member_sha256",
          "installed-wheel": "installed_member_sha256",
          "project-installed": "installed_member_sha256"
        },
        "PY_EXTENSION_LOAD": {
          "cpython-extension": "elf_sha256",
          "installed-extension": "elf_sha256"
        },
        "PY_FROZEN_IMPORT": {
          "cpython-core": "provider_sha256"
        },
        "PY_IMPORT_SOURCE": {
          "cpython-stdlib": "runtime_member_sha256",
          "installed-wheel": "installed_member_sha256",
          "project-installed": "installed_member_sha256"
        },
        "PY_SOURCE_COMPILE": {
          "cpython-stdlib": "runtime_member_sha256",
          "installed-wheel": "installed_member_sha256",
          "project-installed": "installed_member_sha256"
        },
        "PY_TRUSTED_GENERATED_COMPILE": {
          "trusted-generated": "generated_source_sha256"
        },
        "PY_TRUSTED_GENERATED_EXEC": {
          "trusted-generated": "generated_source_sha256"
        }
      },
      "event_tokens": [
        "CPYTHON_CONFIG_READBACK",
        "CTYPES_DLOPEN",
        "DISPATCH_BEGIN",
        "NATIVE_FD_READBACK",
        "PREINIT_NATIVE_IMAGE",
        "PROCESS_EXEC",
        "PROCESS_IMAGE",
        "PY_BUILTIN_IMPORT",
        "PY_CODE_EVAL",
        "PY_CODE_EXEC",
        "PY_EXTENSION_LOAD",
        "PY_FROZEN_IMPORT",
        "PY_IMPORT_SOURCE",
        "PY_SOURCE_COMPILE",
        "PY_TRUSTED_GENERATED_COMPILE",
        "PY_TRUSTED_GENERATED_EXEC"
      ],
      "filter": "Record only authority-process/runtime boundaries and bytes that become executable or interpreted: initial/process images, initial/later native objects, Python builtin/frozen/source/extension imports, file-backed compile/exec/eval, exact trusted bootstrap-generated compile/exec, and dispatch boundary. Ordinary artifact/data/JSON/NPZ/NPY/receipt/index/log/result open/read/stat/enumerate/mmap is excluded unless the same bytes subsequently become executable input.",
      "membership": "Every event resolves to exactly one precommitted authority row and one event-specific origin preimage. Installed origins repeat and equal the exact association preimage; CPython origins equal runtime rows; trusted-generated origins reconstruct one exact standalone discovery/trusted row and consume the nonempty ordered sequence through dedicated adjacent pairs only; native/process origins equal launcher/ELF. Trace and launch instance are evidence only and cannot add or mutate authority.",
      "origin_normalization": {
        "canonical_rule": "No free-form origin string exists. origin is one exact schema_values object with exactly its kind_closed_keys. All path/distribution/version/member/repository/native/generated/readback values are canonical ledger values, and every nullable state is controlled by an explicit tag or kind.",
        "digest_rule": "Every origin uses the sole standalone canonical-JSON-plus-LF preimage above and every association uses temporac.installed-member-association.v4 canonical JSON plus LF.",
        "rejection": "Wrong kind/event/schema, missing/extra key, noncanonical path, member/repository/build-id/association tag mismatch, null/present swap, digest-only substitute, ledger/readback mismatch, event-sha projection mismatch, noncanonical origin reuse, hash collision, or parse/re-encode inequality fails."
      },
      "origin_preimage": {
        "digest_rule": "origin_sha256 is SHA-256 of the exact standalone event-specific origin object serialized as canonical compact sorted-key UTF-8 JSON with exactly one terminal LF. origin is embedded as that parsed object. Verifier reconstructs, canonicalizes, and requires byte/digest equality; string packing, tagged-H, alternate null omission, delimiter concatenation, or digest-only origin is rejected.",
        "event_allowed_kinds": {
          "CPYTHON_CONFIG_READBACK": [
            "runtime-readback"
          ],
          "CTYPES_DLOPEN": [
            "native-image"
          ],
          "DISPATCH_BEGIN": [
            "dispatch-boundary"
          ],
          "NATIVE_FD_READBACK": [
            "runtime-readback"
          ],
          "PREINIT_NATIVE_IMAGE": [
            "native-image",
            "process-image"
          ],
          "PROCESS_EXEC": [
            "process-image"
          ],
          "PROCESS_IMAGE": [
            "process-image"
          ],
          "PY_BUILTIN_IMPORT": [
            "cpython-core"
          ],
          "PY_CODE_EVAL": [
            "cpython-stdlib",
            "installed-wheel",
            "project-installed"
          ],
          "PY_CODE_EXEC": [
            "cpython-core",
            "cpython-stdlib",
            "installed-wheel",
            "project-installed"
          ],
          "PY_EXTENSION_LOAD": [
            "cpython-extension",
            "installed-extension"
          ],
          "PY_FROZEN_IMPORT": [
            "cpython-core"
          ],
          "PY_IMPORT_SOURCE": [
            "cpython-stdlib",
            "installed-wheel",
            "project-installed"
          ],
          "PY_SOURCE_COMPILE": [
            "cpython-stdlib",
            "installed-wheel",
            "project-installed"
          ],
          "PY_TRUSTED_GENERATED_COMPILE": [
            "trusted-generated"
          ],
          "PY_TRUSTED_GENERATED_EXEC": [
            "trusted-generated"
          ]
        },
        "field_equality": {
          "cpython/project/wheel": "all provider/runtime/repository/member paths and hashes repeat the unique precommitted ledger row; no digest-only or path-only membership",
          "installed origins": "association_sha256 resolves one installed row; distribution/version/installed_path/installed_member_sha256/wheel path/member hash/wheel hash and any repository path/hash equal that row and its association preimage exactly",
          "native/process origins": "absolute path, build-id tag/value, ELF hash, launcher/launch-instance values equal the unique ELF/launcher/launch record; native provider_class is exactly initial-native, installed-extension, blas, onednn, openmp, libc, libm, libstdcxx, libpython, dynamic-loader, or supervisor and its installed-association tag/value follows that token.",
          "runtime readback": "readback_schema is exactly temporac.linux-fd-readback.v4 or temporac.cpython-initialization-readback.v4; stage belongs to that schema; readback_bytes is positive, hashes complete canonical bytes including LF, launch instance matches trace, and profile_sha256 equals fd_contract_sha256 or cpython_initialization_profile_sha256 respectively",
          "trusted generated": "trusted set equals exactly the common nonempty two-discovery row array. origin.generator_origin_kind is the sole mapped cpython-stdlib token; origin.generator_origin_sha256 equals row.generator_sha256; origin.trusted_row_sha256 hashes the exact standalone eleven-key row plus LF; every remaining origin field and discovery root reconstruct that unique next row. Only its adjacent dedicated compile-to-exec pair is accepted."
        },
        "kind_closed_keys": {
          "cpython-core": [
            "kind",
            "module_name",
            "provider_kind",
            "provider_path",
            "provider_sha256",
            "runtime_member_normalized_path",
            "schema"
          ],
          "cpython-extension": [
            "absolute_path",
            "build_id_tag",
            "build_id_or_null",
            "elf_sha256",
            "kind",
            "provider_path",
            "runtime_member_normalized_path",
            "runtime_member_sha256",
            "schema"
          ],
          "cpython-stdlib": [
            "kind",
            "normalized_path",
            "provider_path",
            "runtime_member_sha256",
            "schema"
          ],
          "dispatch-boundary": [
            "entrypoint_owner",
            "kind",
            "schema",
            "source_allowlist_sha256"
          ],
          "installed-extension": [
            "absolute_path",
            "association_sha256",
            "build_id_tag",
            "build_id_or_null",
            "distribution",
            "elf_sha256",
            "installed_path",
            "installed_member_sha256",
            "kind",
            "schema",
            "version",
            "wheel_member_path",
            "wheel_member_sha256",
            "wheel_sha256"
          ],
          "installed-wheel": [
            "association_sha256",
            "distribution",
            "installed_path",
            "installed_member_sha256",
            "kind",
            "schema",
            "version",
            "wheel_member_path",
            "wheel_member_sha256",
            "wheel_sha256"
          ],
          "native-image": [
            "absolute_path",
            "build_id_tag",
            "build_id_or_null",
            "elf_sha256",
            "installed_member_association_tag",
            "installed_member_association_sha256_or_null",
            "kind",
            "provider_class",
            "schema"
          ],
          "process-image": [
            "absolute_path",
            "build_id_tag",
            "build_id_or_null",
            "elf_sha256",
            "kind",
            "launch_instance_sha256",
            "launcher_sha256",
            "schema"
          ],
          "project-installed": [
            "association_sha256",
            "distribution",
            "installed_path",
            "installed_member_sha256",
            "kind",
            "repository_path",
            "repository_sha256",
            "schema",
            "version",
            "wheel_member_path",
            "wheel_member_sha256",
            "wheel_sha256"
          ],
          "runtime-readback": [
            "kind",
            "launch_instance_sha256",
            "profile_sha256",
            "readback_bytes",
            "readback_schema",
            "readback_sha256",
            "schema",
            "stage"
          ],
          "trusted-generated": [
            "compile_filename",
            "compile_mode",
            "discovery_output_root_sha256",
            "generated_source_sha256",
            "generator_origin_kind",
            "generator_origin_sha256",
            "generator_qualname",
            "kind",
            "ordinal",
            "profile",
            "schema",
            "source_bytes",
            "trusted_row_sha256"
          ]
        },
        "null_discriminants": {
          "build_id": "build_id_tag is exactly absent or present. absent requires build_id_or_null=null; present requires lowercase nonempty even-length hex equal the unique ELF build-id bytes.",
          "installed row repository": "repository_tag is copied exactly from the association preimage and controls both repository fields; no origin may infer it from path prefix.",
          "installed row wheel": "member_tag is copied exactly from the association preimage; this contract accepts only wheel-member and rejects every installer-generated instance.",
          "native installed association": "installed_member_association_tag is exactly no-installed-member or installed-member. no-installed-member requires the digest null and provider_class not installed-extension; installed-member requires a lowercase digest resolving one association row and provider_class installed-extension."
        },
        "schema_values": {
          "cpython-core": "temporac.executable-origin.cpython-core.v4",
          "cpython-extension": "temporac.executable-origin.cpython-extension.v4",
          "cpython-stdlib": "temporac.executable-origin.cpython-stdlib.v4",
          "dispatch-boundary": "temporac.executable-origin.dispatch-boundary.v4",
          "installed-extension": "temporac.executable-origin.installed-extension.v4",
          "installed-wheel": "temporac.executable-origin.installed-wheel.v4",
          "native-image": "temporac.executable-origin.native-image.v4",
          "process-image": "temporac.executable-origin.process-image.v4",
          "project-installed": "temporac.executable-origin.project-installed.v4",
          "runtime-readback": "temporac.executable-origin.runtime-readback.v4",
          "trusted-generated": "temporac.executable-origin.trusted-generated.v4"
        },
        "trusted_generated_row_projection": {
          "fixed_values": {
            "origin.kind": "trusted-generated",
            "origin.schema": "temporac.executable-origin.trusted-generated.v4",
            "row.generator_origin_class": "cpython-stdlib-source",
            "row.kind": "trusted-bootstrap-generated-source"
          },
          "lookup_and_reconstruction": "Resolve origin.discovery_output_root_sha256 to the exact two-run-equal output committed by the evidence index, then resolve origin.trusted_row_sha256 to exactly one standalone eleven-key trusted row. Recompute the row digest/bytes and generated_source_hex length/hash. Every direct and mapped field below must equal; parse/re-encode equality is mandatory before either event can consume it.",
          "one_to_one_fields": [
            "origin.compile_filename = row.compile_filename",
            "origin.compile_mode = row.compile_mode",
            "origin.generated_source_sha256 = row.generated_source_sha256",
            "origin.generator_qualname = row.generator_qualname",
            "origin.ordinal = row.ordinal",
            "origin.profile = row.profile",
            "origin.source_bytes = row.source_bytes",
            "origin.generator_origin_kind = map(row.generator_origin_class) = cpython-stdlib",
            "origin.generator_origin_sha256 = row.generator_sha256"
          ],
          "origin_only_binding": "origin.discovery_output_root_sha256 equals both evidence-index shared_output_root_sha256 and trusted-index discovery_output_root_sha256; origin.trusted_row_sha256 equals row_hash_preimage. No origin field may be caller supplied or inferred from filename alone.",
          "row_only_validation": "row.generated_source_hex is recovered only from the digest-selected exact row, decodes lowercase even hex to source_bytes, and hashes to generated_source_sha256. row.generator_sha256 resolves one unique cpython-runtime-member stdlib-source provider compatible with compile_filename and generator_qualname.",
          "substitution_failure": "Wrong root/row, alternate encoding, class/kind rename, generator digest rehash, digest collision, source hex omission, ordinal swap, non-adjacent event, generic event token, second consumption or unmapped kind fails."
        },
        "uniqueness": "Association/runtime/generated/ELF authority rows and their identity keys are unique. An origin object is the sole canonical representation of one such identity, so the required source->compile->exec and trusted-compile->exec chains deliberately reuse byte-identical origin objects/digests; any different object with the same digest is a collision failure. Event rows remain unique and gap-free by (process_ordinal,sequence), and reuse outside an event_semantics-required chain fails."
      },
      "schema": "temporac.executable-origin-trace.v4",
      "subprocess_coverage": "Only the exact native supervisor may create an authority child, and the only permitted child image is the same absolute supervisor path. Parent passes authenticated fixed fd roles and allocates the next process_ordinal in PROCESS_EXEC order. Child emits PROCESS_IMAGE sequence zero before embedded-Python initialization and repeats full import-bootstrap/origin validation. Recursive propagation and joined completion are mandatory. Fork-without-exec, shell, PATH lookup, descriptor loss/substitution, missing handshake/image/preinit row, alternate executable, ordinal collision, or unjoined descendant fails."
    },
    "invocation": {
      "bootstrap_rule": "The operating-system entrypoint is the environment-bound native supervisor, not python -m. Its native audit hook exists before CPython initialization. It uses the exact isolated PyConfig/search paths, validates the installed/runtime ledgers, completes the generated-source import bootstrap, then imports and dispatches the one exact callable. No editable/current-tree fallback exists.",
      "command": [
        "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
        "--embedded-cpython-v4"
      ],
      "entrypoint_descriptors": [
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "P00-IMPLEMENT"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "P01-STATIC"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "P02-X0-FIXTURE"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "P03-TOPO-FIXTURE"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "P04-OP-FIXTURE"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "P05-GRAPH-FIXTURE"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "P05M-COUNT-METRIC"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "P06-FRESH-PREFLIGHT"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "S0-COMMIT"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "G0-ACQUIRE"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "K0-FIREWALL"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/teacher/seed=20260815"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/teacher/seed=20260816"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/teacher/seed=20260817"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "G1-TEACHER-AGG"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "K1-COVERAGE"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "G2-CONFORMANCE"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "K2-TARGETS"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/canonical/seed=20260815"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/canonical/seed=20260816"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/canonical/seed=20260817"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/capacity-control/seed=20260815"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/capacity-control/seed=20260816"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/capacity-control/seed=20260817"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260815"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260816"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260817"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/nuisance-only/seed=20260815"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/nuisance-only/seed=20260816"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/nuisance-only/seed=20260817"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260815"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260816"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260817"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/static-code-only/seed=20260815"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/static-code-only/seed=20260816"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/static-code-only/seed=20260817"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/warp-metadata/seed=20260815"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/warp-metadata/seed=20260816"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/warp-metadata/seed=20260817"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/track-length/seed=20260815"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/track-length/seed=20260816"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "temporac.execution.v4/response/shortcut/track-length/seed=20260817"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "G3-RESPONSE-AGG"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "X0I-20260815"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "X0I-20260816"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "X0I-20260817"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "K3-ROUTE"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "K4-DIAG"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "G4-OPERATOR"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "K5-BOUNDARY"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "K6-RESAMPLER"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "NATP-20260815"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "NATP-20260816"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "NATP-20260817"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "G5A-COMMIT"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "G5B-JOIN"
        },
        {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "owner": "K7-FINAL"
        }
      ],
      "owner_inventory": "The exact 57-row entrypoint_descriptors array below is authoritative. Its first 54 owners are exact pre-G5a owner_roster order and final three are G5A-COMMIT, G5B-JOIN, K7-FINAL. No resort, wildcard, alias, duplicate, missing, or extra descriptor.",
      "stdin_closed_keys": [
        "contract_sha256",
        "input_index_bytes",
        "input_index_sha256",
        "owner",
        "schema"
      ],
      "stdin_rule": "fd0, not an argv value, is exactly one canonical JSON object including terminal LF. input_index_bytes is complete canonical stage invocation index encoded as lowercase hex bytes and its digest must match. The supervisor does not read stage payload bytes before DISPATCH_BEGIN. No inherited environment, cwd, path, caller object, or implicit default may add input.",
      "stdin_schema": "temporac.invocation.v4"
    },
    "noncircularity": "Effective contract precedes CPython/native/init/FD/source/wheel inputs; the bootstrap capsule and zero-process precommit precede two fresh discoveries; their exact two-row evidence index/root and equal nonempty output precede trusted-generated rows; those plus code/install indexes precede final environment; launch instances and typed traces follow environment. No capsule depends on discovery evidence, final environment/code/trusted/trace, and no discovery result, trace, or association feeds back into precommit or launch authority.",
    "source_allowlist": {
      "closed_keys": [
        "contract_sha256",
        "entrypoints",
        "rows",
        "schema"
      ],
      "completeness": "Before S0, separately reviewed static closure identifies every repository byte that can execute or be imported by any exact entrypoint and labels it runtime. All remaining installed project payload rows required for install/repository byte equality are installation-equality; mandatory review-only test rows are verifier-test. Runtime rows are exactly the reachable repository closure, not a wildcard; installation-equality and verifier-test rows are forbidden executable origins. protocol_entry.py and native supervisor/fp-control build sources are required future rows. Missing, unlisted-extra runtime origin, wrong purpose, or executed non-runtime row fails.",
      "digest_binding": "code_index_sha256 and existing G5a code_sha256 equal SHA-256 of exact canonical source-allowlist bytes including LF. The index contains no environment or trace digest; installed/runtime indexes instead refer forward to this digest.",
      "entrypoint_descriptor_closed_keys": [
        "argument_schema",
        "callable",
        "owner"
      ],
      "entrypoint_exactness": "The runtime source allowlist entrypoints value is byte-for-byte the exact entrypoint_descriptors array in this amendment. Every callable is pams.temporac.protocol_entry:dispatch and every argument_schema is temporac.invocation.v4; dispatch is reached only after native supervisor initialization and the import bootstrap.",
      "entrypoint_order": "exact entrypoint_descriptors array order: the 54 pre-G5a owner_roster rows, then G5A-COMMIT, G5B-JOIN, K7-FINAL; no resort",
      "future_required_paths": [
        "src/pams/temporac/protocol_entry.py",
        "src/pams/temporac/_fp_control.c",
        "src/pams/temporac/_origin_supervisor.c"
      ],
      "mandatory_minimum_paths": [
        "src/pams/__init__.py",
        "src/pams/temporac/__init__.py",
        "src/pams/temporac/certify.py",
        "src/pams/temporac/contract.py",
        "src/pams/temporac/cue.py",
        "src/pams/temporac/decode.py",
        "src/pams/temporac/evaluator.py",
        "src/pams/temporac/feature_io.py",
        "src/pams/temporac/fixtures.py",
        "src/pams/temporac/gates.py",
        "src/pams/temporac/hashio.py",
        "src/pams/temporac/metrics.py",
        "src/pams/temporac/nola.py",
        "src/pams/temporac/objective.py",
        "src/pams/temporac/prediction.py",
        "src/pams/temporac/preprocess.py",
        "src/pams/temporac/quadrature.py",
        "src/pams/temporac/receipts.py",
        "src/pams/temporac/response.py",
        "src/pams/temporac/runtime.py",
        "src/pams/temporac/teacher.py",
        "src/pams/temporac/training.py",
        "src/pams/temporac/trusted_packer.py",
        "src/pams/temporac/types.py",
        "src/pams/temporac/x0.py",
        "src/pams/types.py",
        "tests/temporac/__init__.py",
        "tests/temporac/test_certificate_artifact.py",
        "tests/temporac/test_certificate_tau.py",
        "tests/temporac/test_contract_hashio.py",
        "tests/temporac/test_feature_io_firewall.py",
        "tests/temporac/test_gates_fixtures.py",
        "tests/temporac/test_metrics_evaluator.py",
        "tests/temporac/test_nola_decode.py",
        "tests/temporac/test_objective_runtime.py",
        "tests/temporac/test_operator_manifest.py",
        "tests/temporac/test_prediction_identity.py",
        "tests/temporac/test_prediction_receipts.py",
        "tests/temporac/test_preprocess_geometry.py",
        "tests/temporac/test_quadrature_cue.py",
        "tests/temporac/test_response_invariants.py",
        "tests/temporac/test_teacher_certificate.py",
        "tests/temporac/test_training_fresh_graph.py",
        "tests/temporac/test_x0.py",
        "tests/temporac/test_x0_manifest.py"
      ],
      "path_rules": "Each repository row is a verified regular non-symlink file whose real path remains under the verified repository root. Normalize only to case-preserving repository-relative POSIX ASCII after realpath verification. Reject non-ASCII, dot segments, alternate separator, symlink, hardlink alias, case alias, duplicate, glob, directory hash, or path not exactly equal to the normalized origin. bytes/hash are exact current-or-future committed file bytes; purpose is one closed token.",
      "purpose_tokens": [
        "installation-equality",
        "runtime",
        "verifier-test"
      ],
      "row_closed_keys": [
        "bytes",
        "path",
        "purpose",
        "sha256"
      ],
      "row_order": "strict bytewise ASCII order of normalized repository-relative POSIX path",
      "schema": "temporac.source-code-allowlist.v4"
    }
  },
  "f4_mapping_composition": {
    "exact_keys": [
      "attack_expected_global_fail_sha256",
      "evaluator_code_sha256",
      "expected_output_sha256",
      "fixture_input_sha256",
      "schema",
      "status"
    ],
    "schema": "temporac.count-metric-fixture-receipt.v1",
    "snapshot_algorithm": [
      "The consumer accepts collections.abc.Mapping directly, including MappingProxyType. Its first operation after the Mapping type check calls items once, exhausts that pass into exactly six pairs, rejects duplicate/missing/extra/non-string keys, and creates plain dict S before any semantic validation.",
      "Inside the same guarded snapshot operation, call items once more into probe T. T must contain the same six unique keys and exact string values as S in fixed key order. Any exception, duplicate, missing/extra key, type drift, or changed value is a TOCTOU failure.",
      "After the stability probe, never read the caller mapping again.",
      "Validate only S, then canonicalize only that same S, then hash/consume those bytes. Validation of the caller object followed by later materialization is forbidden.",
      "consumer(count_metric_fixture_receipt(...)) must work with no caller dict conversion; its bytes equal an ordinary stable dict with the same six pairs."
    ]
  },
  "family_status": "same-family provisional",
  "g5a_and_capability": {
    "consume": "At the non-resumable evaluator process start, atomically consume the one-use grant before any vault access. Persist the consumed state so crash, exception, G5b failure, K7 failure, or process death cannot restore it.",
    "fixed_values": [
      "stub_inclusive_artifact_count is integer 9648",
      "population manifest has exact P402",
      "checkpoint, feature, prediction, target-count, K1-outcome, NATP-completion, pre-G5a receipt-inventory, and executable-origin-trace roots are recomputed from the exact preimages in this amendment",
      "code_sha256 equals exact precommitted temporac.source-code-allowlist.v4 SHA-256",
      "contract_sha256 equals future effective-contract-index SHA-256",
      "environment_sha256 equals exact populated CPU replay environment SHA-256 including active floating-point controls",
      "evaluator_code_sha256 remains the exact P2/S0-bound evaluator implementation byte hash",
      "prediction clean/drift roots are exact merges of the six NATP seed indexes after K6; target_count root contains exactly Cdev and no later subset",
      "receipt_root_sha256 binds the exact twelve-class G5a root index and complete typed DAG, including the 54-owner inventory and NATP K6/prediction lineage"
    ],
    "forbidden": "No training credential, optimizer entry point, score rewrite, vault-derived selection, or evaluator-to-training return path exists.",
    "g5a_schema": "temporac.g5a-receipt.v4",
    "g5b": [
      "Only the same consumed process opens the actual committed vault.",
      "Validate exact vault bytes/root and join commitment, exact bijection with P402, no duplicate/extra/missing identity, manifest-derived component association, positive integer human count and period units, and derive join_cardinality from the verified rows.",
      "G5b PASS requires derived cardinality exactly 402. Any failure burns the grant and forbids retry/resume/restart under it."
    ],
    "k7_order": "K7 begins only after exact G5b PASS in that same consumed fresh process. It locks/probes the CPU state, performs one global 120-checkpoint selection replay, then per-Cdev certificate/target replay, then existing metrics over all 134 development identities including zero-estimate stubs.",
    "preconsume": [
      "Validate only non-vault immutable commitments: effective contract, source allowlist/environment locks, G5a and grant/ledger bytes, P2-METRIC/S0 binding, exact pilot-scope manifest, population P402, all closed indexes/roots/DAG, exact 54-owner receipt inventory, K6-to-three-NATP lineage, exact 9648 prediction product, frozen K1 outcome/Cdev evidence, target counts, and capability invocation_count=1.",
      "Before consumption, do not open, stat, hash, enumerate, deserialize, map, or obtain a filesystem handle for the evaluator vault. A supplied vault row or caller cardinality is not a preconsume input.",
      "If any non-vault check fails, do not consume and do not open the vault."
    ],
    "unchanged_exact_keys": [
      "checkpoint_root_sha256",
      "code_sha256",
      "contract_sha256",
      "environment_sha256",
      "evaluator_code_sha256",
      "feature_root_sha256",
      "population_manifest_sha256",
      "prediction_clean_root_sha256",
      "prediction_drift_root_sha256",
      "receipt_root_sha256",
      "schema",
      "stub_inclusive_artifact_count",
      "target_count_root_sha256",
      "vault_join_commitment_sha256"
    ]
  },
  "generated_at": "2026-08-16T21:55:01+08:00",
  "k1_closed_outcome": {
    "coverage": [
      "All canonical K1 identity/component ratios remain at least ceil(0.8*their frozen G0-eligible denominators) for train and development.",
      "The retained Amendment005 development floors also require at least 108 certified development identities and at least eight certified development components.",
      "Cdev is exactly the ordered val/CERTIFIED projection of this frozen 402-row index. It is not all 402, not necessarily all 134 development identities, and cannot be a later subset.",
      "K2 publishes exactly the target bytes whose hashes were frozen at K1; no regenerated substitute is accepted."
    ],
    "receipt": {
      "closed_keys": [
        "contract_sha256",
        "development_certified_component_count",
        "development_certified_identity_count",
        "outcome_index_bytes",
        "outcome_index_sha256",
        "schema",
        "status",
        "train_certified_component_count",
        "train_certified_identity_count"
      ],
      "exact_key_count": 9,
      "schema": "temporac.k1-certificate-outcome-receipt.v4",
      "status": "PASS only when canonical K1 equal-coverage requirements and the retained development floors are satisfied"
    },
    "row_closed_keys": [
      "certificate_reasons",
      "certificate_status",
      "component_key_hex",
      "eligible",
      "feature_artifact_sha256_or_null",
      "feature_receipt_sha256_or_null",
      "natural_input_receipt_sha256_or_null",
      "opaque_key_hex",
      "population_row_sha256",
      "slot",
      "split",
      "target_artifact_sha256_or_null",
      "target_receipt_sha256_or_null"
    ],
    "row_count": 402,
    "row_hash_preimage": {
      "digest": "k1_outcome_row_sha256=SHA-256 over every standalone byte including the terminal LF.",
      "equality": "Parse the standalone bytes under strict duplicate-key/nonfinite rejection and require the resulting object/value tree byte-canonicalizes exactly to the embedded K1 row. The certified-development foreign key must equal this digest.",
      "exact_bytes": "Construct a standalone JSON object containing exactly the 13 keys in row_closed_keys and the exact values from the committed K1 outcome-index row. Serialize with the amendment canonical JSON rule: UTF-8 no BOM, recursively sorted object keys, compact comma/colon separators, ensure_ascii=false, allow_nan=false, and append exactly one LF byte 0x0a.",
      "forbidden": "No array ordinal, tagged H preimage, embedded-parent bytes, pretty JSON, omitted null, extra key, alternate key order, CRLF, or LF-excluded digest is permitted.",
      "name": "k1_outcome_row_sha256"
    },
    "row_order": "exact population manifest order: train then val, opaque key bytes, slot",
    "row_semantics": [
      "population_row_sha256 hashes the exact standalone canonical eight-key population row bytes including LF; k1_outcome_row_sha256 separately hashes the exact standalone canonical 13-key K1 row bytes including LF as specified by row_hash_preimage.",
      "split, identity, component, eligible, reason codes, and feature_receipt_sha256_or_null equal the same exact population row. When the population feature receipt is nonnull, load it and derive feature_artifact_sha256_or_null from that receipt; when null both feature fields are null.",
      "For eligible=false, certificate_status is NOT_EVALUATED, certificate_reasons equals the frozen nonempty G0/K0 reason vector, natural input is null, and target pair is null.",
      "For eligible=true, certificate_status is ABSTAIN with the exact nonempty certificate reason vector or CERTIFIED with an empty vector. Natural input is null only when the ordered natural inference terminates before teacher output, including LANDMARK; otherwise its exact receipt digest is nonnull.",
      "Target artifact and receipt digests are both nonnull iff status=CERTIFIED; K1 constructs their exact bytes in protected memory and hashes them before K2 publication. Otherwise both are null. This preserves F23 and creates no partial target.",
      "Any population omission, duplication, identity/component/feature swap, later recertification rewrite, or status/null inconsistency fails."
    ],
    "schema": "temporac.k1-certificate-outcome-index.v4",
    "top_level_closed_keys": [
      "contract_sha256",
      "population_manifest_sha256",
      "rows",
      "schema"
    ]
  },
  "k7_target_rederivation": {
    "failure": "Any mismatch is K7 FAIL after capability consumption. The ten-key target receipt is sufficient because the surrounding K1/certified-development/target indexes and typed receipt DAG bind exact feature, natural input, selection, and target bytes; no target receipt schema amendment is made.",
    "per_cdev_steps": [
      "Locate the exact K1 outcome, population, feature, certified-development, and target index rows and require all foreign keys and identities equal.",
      "Load exact feature artifact/receipt and rerun canonical natural inference with the one globally replayed SelectedTeacher; require rebuilt natural-input receipt bytes and phase/reconstruction raw hashes equal.",
      "Construct the typed certificate track internally and run full certify_target. Caller phase, reconstruction, bounds, static code, teacher digest, component, status, or target is never accepted.",
      "Require CERTIFIED, rebuild the unchanged seven-member target NPZ and unchanged ten-key target receipt using the effective contract digest and exact selected checkpoint artifact digest.",
      "Require rebuilt target NPZ bytes and receipt JSON bytes byte-for-byte equal the indexed/K1 artifacts. Digest equality without supplied exact bytes is insufficient."
    ],
    "target_schema_unchanged": {
      "artifact_members": [
        "chi <f8[E]",
        "contract_sha256 |u1[32]",
        "edge_mask |u1[E]",
        "pulse |u1[E]",
        "source_kind |u1[1]",
        "target_mask |u1[E]",
        "teacher_sha256 |u1[32]"
      ],
      "receipt_closed_keys": [
        "artifact_bytes",
        "artifact_sha256",
        "certificate_status",
        "contract_sha256",
        "members",
        "schema",
        "source_key_hex",
        "source_kind",
        "source_unit_index",
        "teacher_sha256"
      ]
    }
  },
  "lineage": {
    "canonical_proposal": {
      "path": "refine-logs/temporac/FINAL_PROPOSAL.md",
      "sha256": "3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391"
    },
    "fixed_pair_paths": [
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE.md",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE.json"
    ],
    "rejected_round1_pair_archived_byte_identically": [
      {
        "bytes": 25001,
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_110212.md",
        "sha256": "fa5ef054de0e243a030905b7a9d0e302aa442cb61b4d39012eaae6dca05a6869"
      },
      {
        "bytes": 24383,
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_110212.json",
        "sha256": "5ea02ae5ccb7ea0f1771222f254ded74318fdb4bc96a5655d5de58a66d7878a2"
      }
    ],
    "rejected_round2_pair_archived_byte_identically": [
      {
        "bytes": 99017,
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_124239.md",
        "sha256": "bd2c0772eb2f8ca70d23c6217b094d58ad69e1167486dbc811b7a4cec9d6e97c"
      },
      {
        "bytes": 74712,
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_124239.json",
        "sha256": "53f3409112550b79b6c11860cf1822e207a8bd600dd9708574ae977548282cd0"
      }
    ],
    "rejected_round3_pair_archived_byte_identically": [
      {
        "bytes": 166625,
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_143304.md",
        "sha256": "8d02c70186c38959c74b774e211427f86435664d00106159c0da11ce5843dae9"
      },
      {
        "bytes": 138394,
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_143304.json",
        "sha256": "187aa3450a9e40f1c75a04b135c5e97e856e18f0165e677a062317e3587be7f2"
      }
    ],
    "rejected_round4_pair_archived_byte_identically": [
      {
        "bytes": 203317,
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_173526.md",
        "sha256": "0552aaca74b34929ead1124cce846d2a583931aa89ad29fcf0f97486b7be0198"
      },
      {
        "bytes": 165291,
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_173526.json",
        "sha256": "04d24ad19b19470bcd675974766a279dc7691d3fdaecf665e3fd24e4a4aaeb82"
      }
    ],
    "rejected_round5_pair_archived_byte_identically": [
      {
        "bytes": 269255,
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_184302.md",
        "sha256": "dbf797e80156d627b14290dfe558222878fdec8ccefbf28a051093130e1349e4"
      },
      {
        "bytes": 226592,
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_184302.json",
        "sha256": "94fc8e5db1031686ea0d5e5646c94606728c18cff19915a6deb231a3bf80cfcb"
      }
    ],
    "round1_fresh_review": [
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW.md",
        "sha256": "35e7f3814759911da256605597a787e86ee5e199ecfded7cc1770a112ccea13d"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW.json",
        "sha256": "5ccd5c66c93f0791c74f5159d9daeaf2b7314c691dd2f25705b4888da565fa2c"
      }
    ],
    "round1_reviewer_trace": [
      {
        "path": ".aris/traces/research-refine/20260816_temporac_certificate_amendment005_review_sol/meta.json",
        "sha256": "21a37e19089b7b6f3af50ff0618d3b111165ae8822b4da739c2e997b3c2e73da"
      },
      {
        "path": ".aris/traces/research-refine/20260816_temporac_certificate_amendment005_review_sol/request.json",
        "sha256": "d164f82b8ca544c465791daaedf41f40deceb138c339943b7b255d62e4bd348b"
      },
      {
        "path": ".aris/traces/research-refine/20260816_temporac_certificate_amendment005_review_sol/response.md",
        "sha256": "35e7f3814759911da256605597a787e86ee5e199ecfded7cc1770a112ccea13d"
      }
    ],
    "round1_round6_reviewer_trace_aggregator": [
      {
        "path": ".aris/traces/research-refine/20260816_temporac_certificate_amendment005_review_sol/run.meta.json",
        "sha256": "7a860dc5a9be3878aab060e541430a5731237fc1b2e9b75d7798100a0a3a1da2"
      }
    ],
    "round2_fresh_review": [
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND2.md",
        "sha256": "9d401fcde35786bfe2925cf31a803bd515668a156f08ae5bc9ef5c27810a4101"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND2.json",
        "sha256": "ed877ba3d8fd15c76c9908dd694bfbb804ed0c7b05f99a97853dc55c51ea56bd"
      }
    ],
    "round2_versioned_pair_paths": [
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_121046.md",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_121046.json"
    ],
    "round3_fresh_review": [
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND3.md",
        "sha256": "e29f1cab1e92feba575cf7d5f446de4f8388b42b7c2d1c65e21d8187f856a54e"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND3.json",
        "sha256": "df7917f5eb215779afca36a4db1b978f0a2f6b64b720267b2b14cb20e703339a"
      }
    ],
    "round3_versioned_pair_paths": [
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_141214.md",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_141214.json"
    ],
    "round4_fresh_review": [
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND4.md",
        "sha256": "988eb7c2b61eb445a07e1c17b7bcf44be256f9948d6600a6ed6edc62ef71cfc1"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND4.json",
        "sha256": "f8552023abb52c98baec6a9143d15e93e3268d883bf3967c446c3fcbcd9d06d9"
      }
    ],
    "round4_review_inputs": [
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND3.md",
        "sha256": "e29f1cab1e92feba575cf7d5f446de4f8388b42b7c2d1c65e21d8187f856a54e"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND3.json",
        "sha256": "df7917f5eb215779afca36a4db1b978f0a2f6b64b720267b2b14cb20e703339a"
      }
    ],
    "round4_versioned_pair_paths": [
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_154812.md",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_154812.json"
    ],
    "round5_fresh_review": [
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND5.md",
        "sha256": "ca3807657ee4197996c49c270cabd73bb43e439422173ca780849f433113a6f7"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND5.json",
        "sha256": "8f2490fcfe8434e1d3f2adc3a2a5db7afb4166302f421fa06e45512b764cf2ff"
      }
    ],
    "round5_review_inputs": [
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND4.md",
        "sha256": "988eb7c2b61eb445a07e1c17b7bcf44be256f9948d6600a6ed6edc62ef71cfc1"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND4.json",
        "sha256": "f8552023abb52c98baec6a9143d15e93e3268d883bf3967c446c3fcbcd9d06d9"
      }
    ],
    "round5_versioned_pair_paths": [
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_173527.md",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_173527.json"
    ],
    "round6_accepted_pair_archived_byte_identically": [
      {
        "bytes": 312728,
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_215500.md",
        "sha256": "2eaa3cfb746804fdc5890fec4c252c877a031a4abedef7b9ef6eeac0a4839395"
      },
      {
        "bytes": 261196,
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_215500.json",
        "sha256": "638efc8708f74c8514247f4f4c5e2311595e90edc6744c9b371886f0789af164"
      }
    ],
    "round6_review_inputs": [
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND5.md",
        "sha256": "ca3807657ee4197996c49c270cabd73bb43e439422173ca780849f433113a6f7"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND5.json",
        "sha256": "8f2490fcfe8434e1d3f2adc3a2a5db7afb4166302f421fa06e45512b764cf2ff"
      }
    ],
    "round6_versioned_pair_paths": [
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_184303.md",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_184303.json"
    ],
    "round7_review_inputs": [
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND6.md",
        "sha256": "4a3199c3ac18e3fd2a5bb7ecf6d66e10bc4a5cebb82363803c47867234b70e67"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND6.json",
        "sha256": "fcf8a9d4949f6b456f18bfb4591c512b6628c6ec7eb5a055873a995992f46ad6"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_WAVE1_CONTRACT_IMPLEMENTATION_POSTFIX_REVIEW_20260816.md",
        "sha256": "7cf1f1a013932a7f9317f094d28dc54af9ab0da7adb6caab4a5964941231fa3b"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_WAVE1_CONTRACT_IMPLEMENTATION_POSTFIX_REVIEW_20260816.json",
        "sha256": "bdec8e1f13023e47f2be85010b7cdae09a1e219c106956a24980295bc9ce533e"
      }
    ],
    "round7_versioned_pair_paths": [
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_215501.md",
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_215501.json"
    ]
  },
  "natural_inference": {
    "builder_inputs": [
      "validated typed FeatureRecord plus exact feature artifact/receipt bytes",
      "one typed SelectedTeacher created by the global replay"
    ],
    "exact_order": [
      "Validate feature artifact, feature receipt, identity, source binding, and exact population-manifest row.",
      "Run canonical preprocess_identity once. teacher_input is exact C-order <f4[T,215]; continuous is columns 0:149 converted once to C-order <f8[T,149]; use the same canonical geometry, clocks, masks, run bounds, and binary64 edge lengths.",
      "For every run in run order, before any teacher call, run the geometry-only two-stage integer_landmarks rule. If any run has fewer than three retained landmarks, immediately return ABSTAIN reasons [11] and create no natural-certificate-input receipt, teacher-output artifact, target artifact, or target receipt.",
      "Within the same run, consecutive retained landmarks alone form ordered traversal edge bounds [L_j,L_(j+1)); no cross-run or nonconsecutive bound.",
      "For each run independently compute F5 from continuous[149], edge_lengths, and these bounds. Visit traversals in order and subedges left-to-right; use normative Neumaier for every denominator and numerator. Mean term is lambda*((A+B)/2); second moment is lambda*(((A*A+A*B)+B*B)/3) in the written binary64 order. Equal-average traversal means/seconds with Neumaier; s=sqrt(max(qbar-mbar*mbar,+0.0)); concatenate [mbar,s] C-order <f8[298].",
      "Cast each complete run static vector once to C-order <f4[298]. In run order issue one CPU float32 TempoRACTeacher forward for that run's exact <f4[Trun,215] input; no batch or padding. Use the exact environment, eval, inference_mode, one thread, no autocast/TF32.",
      "Validate finite float32 outputs, place them in canonical run/sample order, and cast each complete assembled C-order output exactly once to <f8[T,2] and <f8[T,149]. Every retained sample must belong to exactly one run; overlap, uncovered retained sample, or synthetic fill fails.",
      "Hash exact raw C-order float64 output bytes, build the unchanged natural-certificate-input receipt, bind the typed certificate track, run full certify_target, and either emit no target on ABSTAIN or the unchanged target NPZ/receipt on CERTIFIED."
    ],
    "f5_definition": "This section is the unique natural F5 interface. Current helper names are not authority; any implementation using ordinary np.sum, pairwise/reordered reduction, identity-wide static pooling, teacher-derived landmarks, or caller bounds is nonconforming.",
    "forbidden_parameters": [
      "selected_teacher_sha256",
      "phase",
      "reconstruction",
      "static_code",
      "traversal_bounds",
      "checkpoint digest without artifact",
      "caller module or state_dict"
    ],
    "natural_input_receipt": {
      "closed_keys": [
        "continuous_mask_sha256",
        "continuous_sha256",
        "feature_artifact_sha256",
        "feature_receipt_sha256",
        "geometry_sha256",
        "opaque_key_hex",
        "phase_sha256",
        "reconstruction_sha256",
        "run_bounds_sha256",
        "sampled_source_clock_sha256",
        "schema",
        "slot",
        "source_binding_sha256",
        "teacher_input_sha256",
        "teacher_sha256"
      ],
      "exact_key_count": 15,
      "schema": "temporac.natural-certificate-input.v4",
      "teacher_binding": "teacher_sha256 is SHA-256 of exact selected checkpoint artifact bytes; phase/reconstruction hashes are post-cast float64 raw bytes; static vectors and traversal bounds are independently rederived and are not caller evidence"
    },
    "output_type": "NaturalTeacherOutput is an opaque immutable package-private value built only by the canonical builder and contains the post-cast phase/reconstruction arrays, their rebuilt natural-input receipt bytes, and the SelectedTeacher provenance handle; no public constructor accepts arrays or digests."
  },
  "pre_g5a_receipt_inventory": {
    "aggregate": "inventory_sha256=SHA-256(exact canonical inventory bytes including LF); the bytes/hash are one typed G5a root-index row",
    "class_counts": {
      "g1-teacher-selection": 1,
      "k1-certificate-outcome": 1,
      "k3": 1,
      "k4": 1,
      "natural-prediction-completion": 3,
      "pre-g5a-stage": 17,
      "run": 27,
      "x0-inference": 3
    },
    "exact_roster_direct_edge_count": 106,
    "g0_payload_specialization": {
      "closed_keys": [
        "contract_sha256",
        "population_manifest_bytes",
        "population_manifest_sha256",
        "schema"
      ],
      "equality": "The G0-ACQUIRE generic receipt payload_index_sha256 hashes exact canonical bytes of this object. Its population manifest bytes/hash equal the independently supplied exact P402 manifest and K1 outcome-index population_manifest_sha256; byte count is positive and hashes complete bytes.",
      "schema": "temporac.g0-acquisition-payload-index.v4"
    },
    "owner_roster": [
      {
        "node_class": "pre-g5a-stage",
        "owner": "P00-IMPLEMENT",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": []
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "P01-STATIC",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "P00-IMPLEMENT"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "P02-X0-FIXTURE",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "P01-STATIC"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "P03-TOPO-FIXTURE",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "P02-X0-FIXTURE"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "P04-OP-FIXTURE",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "P03-TOPO-FIXTURE"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "P05-GRAPH-FIXTURE",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "P04-OP-FIXTURE"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "P05M-COUNT-METRIC",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "P01-STATIC"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "P06-FRESH-PREFLIGHT",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "P01-STATIC",
          "P05-GRAPH-FIXTURE",
          "P05M-COUNT-METRIC"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "S0-COMMIT",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "P00-IMPLEMENT",
          "P01-STATIC",
          "P02-X0-FIXTURE",
          "P03-TOPO-FIXTURE",
          "P04-OP-FIXTURE",
          "P05-GRAPH-FIXTURE",
          "P05M-COUNT-METRIC",
          "P06-FRESH-PREFLIGHT"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "G0-ACQUIRE",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "S0-COMMIT"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "K0-FIREWALL",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "G0-ACQUIRE"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/teacher/seed=20260815",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "S0-COMMIT",
          "K0-FIREWALL"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/teacher/seed=20260816",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "S0-COMMIT",
          "K0-FIREWALL"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/teacher/seed=20260817",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "S0-COMMIT",
          "K0-FIREWALL"
        ]
      },
      {
        "node_class": "g1-teacher-selection",
        "owner": "G1-TEACHER-AGG",
        "receipt_schema": "temporac.g1-teacher-selection-receipt.v4",
        "upstream_owner_tokens": [
          "temporac.execution.v4/teacher/seed=20260815",
          "temporac.execution.v4/teacher/seed=20260816",
          "temporac.execution.v4/teacher/seed=20260817"
        ]
      },
      {
        "node_class": "k1-certificate-outcome",
        "owner": "K1-COVERAGE",
        "receipt_schema": "temporac.k1-certificate-outcome-receipt.v4",
        "upstream_owner_tokens": [
          "G0-ACQUIRE",
          "G1-TEACHER-AGG"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "G2-CONFORMANCE",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "P05-GRAPH-FIXTURE",
          "K1-COVERAGE"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "K2-TARGETS",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "G1-TEACHER-AGG",
          "K1-COVERAGE",
          "G2-CONFORMANCE"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/canonical/seed=20260815",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/canonical/seed=20260816",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/canonical/seed=20260817",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/capacity-control/seed=20260815",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/capacity-control/seed=20260816",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/capacity-control/seed=20260817",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260815",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260816",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260817",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/nuisance-only/seed=20260815",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/nuisance-only/seed=20260816",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/nuisance-only/seed=20260817",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260815",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260816",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260817",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/static-code-only/seed=20260815",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/static-code-only/seed=20260816",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/static-code-only/seed=20260817",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/warp-metadata/seed=20260815",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/warp-metadata/seed=20260816",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/warp-metadata/seed=20260817",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/track-length/seed=20260815",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/track-length/seed=20260816",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "run",
        "owner": "temporac.execution.v4/response/shortcut/track-length/seed=20260817",
        "receipt_schema": "temporac.run-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "G3-RESPONSE-AGG",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "K2-TARGETS",
          "temporac.execution.v4/response/canonical/seed=20260815",
          "temporac.execution.v4/response/canonical/seed=20260816",
          "temporac.execution.v4/response/canonical/seed=20260817",
          "temporac.execution.v4/response/capacity-control/seed=20260815",
          "temporac.execution.v4/response/capacity-control/seed=20260816",
          "temporac.execution.v4/response/capacity-control/seed=20260817",
          "temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260815",
          "temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260816",
          "temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260817",
          "temporac.execution.v4/response/shortcut/nuisance-only/seed=20260815",
          "temporac.execution.v4/response/shortcut/nuisance-only/seed=20260816",
          "temporac.execution.v4/response/shortcut/nuisance-only/seed=20260817",
          "temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260815",
          "temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260816",
          "temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260817",
          "temporac.execution.v4/response/shortcut/static-code-only/seed=20260815",
          "temporac.execution.v4/response/shortcut/static-code-only/seed=20260816",
          "temporac.execution.v4/response/shortcut/static-code-only/seed=20260817",
          "temporac.execution.v4/response/shortcut/warp-metadata/seed=20260815",
          "temporac.execution.v4/response/shortcut/warp-metadata/seed=20260816",
          "temporac.execution.v4/response/shortcut/warp-metadata/seed=20260817",
          "temporac.execution.v4/response/shortcut/track-length/seed=20260815",
          "temporac.execution.v4/response/shortcut/track-length/seed=20260816",
          "temporac.execution.v4/response/shortcut/track-length/seed=20260817"
        ]
      },
      {
        "node_class": "x0-inference",
        "owner": "X0I-20260815",
        "receipt_schema": "temporac.x0-inference-receipt.v4",
        "upstream_owner_tokens": [
          "G3-RESPONSE-AGG"
        ]
      },
      {
        "node_class": "x0-inference",
        "owner": "X0I-20260816",
        "receipt_schema": "temporac.x0-inference-receipt.v4",
        "upstream_owner_tokens": [
          "G3-RESPONSE-AGG"
        ]
      },
      {
        "node_class": "x0-inference",
        "owner": "X0I-20260817",
        "receipt_schema": "temporac.x0-inference-receipt.v4",
        "upstream_owner_tokens": [
          "G3-RESPONSE-AGG"
        ]
      },
      {
        "node_class": "k3",
        "owner": "K3-ROUTE",
        "receipt_schema": "temporac.k3-receipt.v4",
        "upstream_owner_tokens": [
          "X0I-20260815",
          "X0I-20260816",
          "X0I-20260817"
        ]
      },
      {
        "node_class": "k4",
        "owner": "K4-DIAG",
        "receipt_schema": "temporac.k4-receipt.v4",
        "upstream_owner_tokens": [
          "K3-ROUTE",
          "X0I-20260815",
          "X0I-20260816",
          "X0I-20260817"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "G4-OPERATOR",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "P04-OP-FIXTURE",
          "K4-DIAG",
          "X0I-20260815",
          "X0I-20260816",
          "X0I-20260817"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "K5-BOUNDARY",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "G4-OPERATOR"
        ]
      },
      {
        "node_class": "pre-g5a-stage",
        "owner": "K6-RESAMPLER",
        "receipt_schema": "temporac.pre-g5a-stage-receipt.v4",
        "upstream_owner_tokens": [
          "G1-TEACHER-AGG",
          "G4-OPERATOR",
          "K5-BOUNDARY"
        ]
      },
      {
        "node_class": "natural-prediction-completion",
        "owner": "NATP-20260815",
        "receipt_schema": "temporac.natural-prediction-completion-receipt.v4",
        "upstream_owner_tokens": [
          "K6-RESAMPLER"
        ]
      },
      {
        "node_class": "natural-prediction-completion",
        "owner": "NATP-20260816",
        "receipt_schema": "temporac.natural-prediction-completion-receipt.v4",
        "upstream_owner_tokens": [
          "K6-RESAMPLER"
        ]
      },
      {
        "node_class": "natural-prediction-completion",
        "owner": "NATP-20260817",
        "receipt_schema": "temporac.natural-prediction-completion-receipt.v4",
        "upstream_owner_tokens": [
          "K6-RESAMPLER"
        ]
      }
    ],
    "receipt_schemas": {
      "temporac.g1-teacher-selection-receipt.v4": {
        "closed_keys": [
          "code_index_sha256",
          "contract_sha256",
          "environment_sha256",
          "schema",
          "score_index_sha256",
          "selection_receipt_sha256",
          "status",
          "teacher_checkpoint_evidence_index_sha256",
          "teacher_run_receipt_sha256",
          "teacher_tune_input_receipt_sha256"
        ],
        "exact_key_count": 10,
        "rules": "dedicated G1 owner; teacher_run_receipt_sha256 exact length-three array in seed order"
      },
      "temporac.k1-certificate-outcome-receipt.v4": {
        "closed_keys": [
          "contract_sha256",
          "development_certified_component_count",
          "development_certified_identity_count",
          "outcome_index_bytes",
          "outcome_index_sha256",
          "schema",
          "status",
          "train_certified_component_count",
          "train_certified_identity_count"
        ],
        "exact_key_count": 9,
        "rules": "dedicated K1 owner; unchanged nine-key receipt; inventory k1.upstream ordinals bind G0 P402 population and G1 selection evidence exactly as receipt_dag.k1_upstream_validation"
      },
      "temporac.k3-receipt.v4": {
        "byte_field_rule": "payload_index_bytes is a positive JSON integer equal to the complete separately supplied canonical payload-index byte count; payload_index_sha256 hashes those complete bytes including LF.",
        "closed_keys": [
          "code_index_sha256",
          "contract_sha256",
          "environment_sha256",
          "owner",
          "payload_index_bytes",
          "payload_index_sha256",
          "schema",
          "status",
          "upstream_receipt_sha256"
        ],
        "exact_key_count": 9,
        "rules": "owner K3-ROUTE; upstream exactly X0I-20260815/16/17; payload binds complete F15/F18/bootstrap evidence"
      },
      "temporac.k4-receipt.v4": {
        "byte_field_rule": "payload_index_bytes is a positive JSON integer equal to the complete separately supplied canonical payload-index byte count; payload_index_sha256 hashes those complete bytes including LF.",
        "closed_keys": [
          "code_index_sha256",
          "contract_sha256",
          "environment_sha256",
          "owner",
          "payload_index_bytes",
          "payload_index_sha256",
          "schema",
          "status",
          "upstream_receipt_sha256"
        ],
        "exact_key_count": 9,
        "rules": "owner K4-DIAG; upstream K3 then X0I-20260815/16/17; payload binds complete F16/F17/bootstrap evidence"
      },
      "temporac.natural-prediction-completion-receipt.v4": {
        "closed_keys": [
          "clean_seed_index_sha256",
          "clean_seed_root_sha256",
          "code_index_sha256",
          "contract_sha256",
          "drift_seed_index_sha256",
          "drift_seed_root_sha256",
          "environment_sha256",
          "k6_receipt_sha256",
          "owner",
          "pilot_scope_manifest_sha256",
          "schema",
          "seed",
          "status",
          "upstream_receipt_sha256"
        ],
        "exact_key_count": 14,
        "rules": "owner is exact NATP-SEED and seed agrees; upstream_receipt_sha256 is an exact length-one lowercase digest array; element zero equals k6_receipt_sha256 and the unique K6-RESAMPLER inventory receipt digest; both condition seed indexes/roots and exact pilot-scope manifest are mandatory; PASS creates no metric or gate authority"
      },
      "temporac.pre-g5a-stage-receipt.v4": {
        "byte_field_rule": "payload_index_bytes is a positive JSON integer equal to the complete separately supplied canonical payload-index byte count; payload_index_sha256 hashes those complete bytes including LF.",
        "closed_keys": [
          "code_index_sha256",
          "contract_sha256",
          "environment_sha256_or_null",
          "owner",
          "payload_index_bytes",
          "payload_index_sha256",
          "schema",
          "status",
          "upstream_receipt_sha256"
        ],
        "exact_key_count": 9,
        "rules": "owner is one exact roster token; payload index byte count/hash and supplied canonical bytes agree; upstream array equals roster. environment is null exactly for P00 through P05M and is the exact populated environment digest for P06 and every later generic stage. PASS requires stage-specific effective-plan validation."
      },
      "temporac.run-receipt.v4": {
        "closed_keys": [
          "code_sha256",
          "config_sha256",
          "contract_sha256",
          "environment_sha256",
          "final_step",
          "job_name_hex",
          "optimizer_sha256",
          "ordered_checkpoint_sha256",
          "resource",
          "rng_sha256",
          "schema",
          "seed",
          "source_cycle_sha256",
          "status",
          "upstream_receipt_sha256"
        ],
        "exact_key_count": 15,
        "rules": "unchanged proposal schema; ordered arrays are lowercase digests. Teacher upstream order is S0 then K0; every response upstream array is K2 only."
      },
      "temporac.x0-inference-receipt.v4": {
        "closed_keys": [
          "code_index_sha256",
          "contract_sha256",
          "environment_sha256",
          "owner",
          "prediction_index_sha256",
          "schema",
          "seed",
          "status",
          "upstream_receipt_sha256"
        ],
        "exact_key_count": 9,
        "rules": "owner X0I-SEED and seed agree; exactly one G3 upstream; prediction index is the exact seed synthetic shard required by K3/K4"
      }
    },
    "row_closed_keys": [
      "node_class",
      "owner",
      "receipt_schema",
      "receipt_sha256",
      "upstream_owner_tokens",
      "upstream_receipt_sha256"
    ],
    "row_count": 54,
    "row_order": "exact owner_roster order below; no ASCII resort or filesystem discovery",
    "row_rules": [
      "Each runtime row contains exactly the six closed keys. receipt_sha256 is exact canonical receipt-byte digest. upstream_receipt_sha256 is an exact digest array with length and ordinal matching upstream_owner_tokens.",
      "For temporac.pre-g5a-stage-receipt.v4 and X0I/K3/K4 receipts, payload owner equals row owner and payload upstream_receipt_sha256 equals the row array. For NATP, its new exact length-one upstream_receipt_sha256 equals the row array and element zero byte-equals both payload.k6_receipt_sha256 and the K6-RESAMPLER receipt digest. For run receipts, decoded job_name_hex/seed identify the unique owner and payload upstream array equals row array.",
      "G1 is uniquely identified by its dedicated schema; teacher_run_receipt_sha256 is exactly three teacher-run row digests in roster/seed order. K1 is uniquely identified by its dedicated schema; its inventory row array is exact [G0-ACQUIRE receipt,G1-TEACHER-AGG receipt], and the k1.upstream validation rules bind those direct digests to the population and selected-teacher evidence.",
      "P0/P1/P2/P2-METRIC/P3 are proposal prose aliases only and forbidden as owner values. Exact mapping is P0=P00-IMPLEMENT, P1=P01-STATIC, P2 fixture=P02-X0-FIXTURE/P03-TOPO-FIXTURE/P04-OP-FIXTURE/P05-GRAPH-FIXTURE, P2-METRIC=P05M-COUNT-METRIC, P3=P06-FRESH-PREFLIGHT.",
      "P05M generic payload index must bind exact Amendment003 candidate/review/runner/inner-summary evidence required by effective plan; this inventory neither repairs its separate REVISE disposition nor creates PASS authority."
    ],
    "schema": "temporac.pre-g5a-receipt-inventory.v4",
    "top_level_closed_keys": [
      "contract_sha256",
      "rows",
      "schema"
    ]
  },
  "receipt_dag": {
    "acyclicity": "Edges point from a receipt to an already-existing dependency. Topological sort over exact nodes must consume every node. The 54-owner roster generates exactly 106 direct edges and fixes F23 order X0Ix3 -> K3 -> K4 -> G4 and K6 -> NATPx3. K1 points only backward to G0/G1. NATP points only backward to K6 and prediction receipts already frozen for that NATP completion. The containing G5a object/root cannot be a node or dependency.",
    "completeness": "Expected nodes and edges are mechanically derived from exact artifact/index rows and exact 54-row roster. Every receipt byte payload is supplied and hash/closed-schema validated. Edge ordinal equals source payload/index occurrence. NATP exact14 and K1 dedicated inventory rules are schema-aware. Any missing, extra, duplicate, wrong class/owner/schema, payload-to-owner mismatch, dangling reference, multiplicity/ordinal mismatch, or cycle fails.",
    "edge_closed_keys": [
      "from_receipt_sha256",
      "ordinal",
      "role",
      "to_receipt_sha256"
    ],
    "edge_order": "decoded from digest, ASCII role, numeric ordinal, decoded to digest",
    "exact_class_multiplicity": [
      "teacher-tune-input: 1",
      "teacher-checkpoint: 120",
      "teacher-tune-evaluation: 120",
      "teacher-selection: 1",
      "g1-teacher-selection: 1",
      "k1-certificate-outcome: 1",
      "run: exactly 27, three teacher then 24 response owners from the inventory",
      "x0-inference: exactly 3, owners X0I-20260815/16/17",
      "k3: exactly 1, owner K3-ROUTE",
      "k4: exactly 1, owner K4-DIAG",
      "natural-prediction-completion: exactly 3, owners NATP-20260815/16/17",
      "pre-g5a-stage: exactly 17 owner rows from the inventory",
      "prediction: exactly 9648",
      "feature: exactly the nonnull feature receipts in the 402-row feature index",
      "natural-certificate-input: exactly the nonnull natural input receipts in the 402-row K1 outcome index",
      "target: exactly all rows of the target index"
    ],
    "explicit_exclusions": [
      "the containing temporac.g5a-receipt.v4 receipt",
      "the G5a root index itself",
      "capability-request ledger record or capability grant",
      "G5b receipt or K7 receipt",
      "metric payload or pilot report",
      "any post-G5a, future, retry, result, or acceptance-review receipt"
    ],
    "k1_upstream_validation": {
      "g0_equality": "Validate exact G0 receipt bytes/schema/owner, load the exact g0-acquisition-payload-index bytes named by its payload_index bytes/hash, then require its exact P402 population_manifest bytes/hash equal both the separately root-bound population manifest and k1-certificate-outcome-index.population_manifest_sha256. Every K1 row population_row_sha256 is an exact row of that same manifest.",
      "g1_equality": "Validate exact G1 receipt bytes/schema/owner; its selection, tune-input, score-index, checkpoint-evidence, and three seed-ordered run digests resolve to exact DAG nodes. Every nonnull K1 natural-input receipt names the same G1 selection receipt and selected checkpoint artifact/receipt; every K1 target association inherits that natural input. Any empty-evidence special case still retains the same direct G1 digest.",
      "inventory_equality": "The K1 inventory row upstream_owner_tokens is exactly [G0-ACQUIRE,G1-TEACHER-AGG] and upstream_receipt_sha256 is exactly the corresponding two validated digests. K1's nine-key receipt remains unchanged; the inventory is its canonical direct-dependency container.",
      "ordinals": [
        "ordinal 0 role k1.upstream points from K1-COVERAGE to the unique G0-ACQUIRE inventory receipt",
        "ordinal 1 role k1.upstream points from K1-COVERAGE to the unique G1-TEACHER-AGG inventory receipt"
      ]
    },
    "node_class_tokens": [
      "feature",
      "g1-teacher-selection",
      "k1-certificate-outcome",
      "k3",
      "k4",
      "natural-certificate-input",
      "natural-prediction-completion",
      "pre-g5a-stage",
      "prediction",
      "run",
      "target",
      "teacher-checkpoint",
      "teacher-selection",
      "teacher-tune-evaluation",
      "teacher-tune-input",
      "x0-inference"
    ],
    "node_closed_keys": [
      "class",
      "owner_key",
      "receipt_sha256"
    ],
    "node_order": "ASCII class, ASCII owner_key, then decoded raw receipt digest",
    "owner_key_encoding": "owner_key is lowercase H_hex(temporac.receipt-owner.v4, class ASCII, then the exact typed owner fields below); it is never a slash-joined or caller string",
    "owner_key_rules": [
      "feature and natural input: class, split ASCII, raw32 opaque key, uint64_be(slot)",
      "teacher checkpoint and tune evaluation: class, uint64_be(seed), uint32_be(step)",
      "run: class, exact decoded job_name_hex bytes, uint64_be(seed); job name must be one exact inventory owner",
      "prediction: class, split ASCII, raw32 opaque key, uint64_be(slot), arm ASCII, uint64_be(seed), condition ASCII",
      "target: class, uint8 source_kind, raw32 source key, uint64_be(source_unit_index)",
      "x0-inference and natural-prediction-completion: class, uint64_be(seed)",
      "pre-g5a-stage, G1, K1, K3, and K4: class then exact inventory owner ASCII",
      "teacher tune input and selection: class then exact schema ASCII"
    ],
    "required_edges": [
      "Every teacher checkpoint -> its one teacher run with checkpoint.run ordinal 0.",
      "Every tune evaluation -> checkpoint ordinal 0 and tune input ordinal 0 under distinct roles.",
      "Teacher selection -> three per-seed winner checkpoint receipts with ordinals 0,1,2 in seed order, and -> tune input ordinal 0.",
      "G1 -> selection ordinal 0, tune input ordinal 0, and three teacher runs with g1.teacher-run ordinals 0,1,2 in seed order. Receipt/evidence arrays equal those same three digests.",
      "Each of 27 run receipts -> every ordered upstream_receipt_sha256 element using run.upstream and its zero-based payload-array ordinal. Teacher arrays are S0,K0; response arrays are K2.",
      "Each pre-g5a-stage, x0-inference, K3, and K4 receipt -> every inventory upstream digest using stage.upstream and roster-array ordinal. G1 teacher dependencies instead use g1.teacher-run.",
      "K1 -> G0-ACQUIRE and G1-TEACHER-AGG using k1.upstream ordinals 0 and 1 exactly, with all k1_upstream_validation equalities.",
      "Every natural input -> feature and teacher selection, each ordinal 0.",
      "Every target -> its natural input when natural and teacher selection, each ordinal 0.",
      "Every prediction -> feature and owning response run, each ordinal 0; manifest component is separately root-bound non-receipt association.",
      "K1 -> every K1-referenced feature, natural input, and target with ordinal equal that receipt's zero-based occurrence in exact K1 row order within its role.",
      "Each NATP receipt -> its exact length-one payload upstream_receipt_sha256[0] using natp.k6 ordinal 0; that digest equals payload.k6_receipt_sha256 and unique K6-RESAMPLER inventory receipt.",
      "Each NATP completion -> exactly 3216 prediction receipts for its seed using natp.prediction ordinals 0..3215: clean seed-index order first, then drift seed-index order.",
      "These rules generate exactly all 106 roster-direct dependency edges: 58 generic pre-g5a stage.upstream, 30 run.upstream, three g1.teacher-run, two k1.upstream, three X0I stage.upstream, three K3 stage.upstream, four K4 stage.upstream, and three natp.k6. Thus the aggregate stage.upstream role count is 68. No catch-all or transitive-only edge is permitted."
    ],
    "role_tokens": [
      "checkpoint.run",
      "evaluation.checkpoint",
      "evaluation.tune-input",
      "g1.selection",
      "g1.teacher-run",
      "g1.tune-input",
      "k1.feature",
      "k1.natural-input",
      "k1.target",
      "k1.upstream",
      "natp.k6",
      "natp.prediction",
      "natural-input.feature",
      "natural-input.selection",
      "prediction.feature",
      "prediction.run",
      "run.upstream",
      "selection.per-seed-winner",
      "selection.tune-input",
      "stage.upstream",
      "target.natural-input",
      "target.selection"
    ],
    "schema": "temporac.receipt-dag.v4",
    "top_level_closed_keys": [
      "contract_sha256",
      "edges",
      "nodes",
      "schema"
    ]
  },
  "required_test_contract": {
    "mapping_to_official_findings": {
      "A005-R2-B1": "exact normalized SmoothL1 binary64 piecewise DAG and boundary; one-run <i4[[0,N_v]] certificate input; ten traversal bounds analytic/F11-only; exact three-run digest arrays",
      "A005-R2-B2": "active FE_TONEAREST/MXCSR/denormal/backend state lock, exact probes and fresh-process fail-closed replay",
      "A005-R2-B3": "acyclic precommitted source-byte allowlist plus independent executable-origin traces with exact entrypoints, event filter, normalization, and subprocess propagation",
      "A005-R2-B4": "closed 54-owner receipt inventory and numeric typed DAG containing X0Ix3, K3, K4, every gate/fixture/run/NATP owner, schema, multiplicity, and direct order",
      "A005-R2-B5": "run.upstream ordinal edges and exact G1/evidence three-teacher-run container/order",
      "A005-R2-B6": "exact pilot-scope/NATP completion index, six seed indexes/roots, K6 lineage, final 9648 merge, and G5a root/DAG commitment",
      "A005-R2-B7": "standalone exact canonical 13-key K1-row-plus-LF preimage for k1_outcome_row_sha256",
      "A005-R2-B8": "the original 50-path Round3 rehash remains exact; Round4 additionally binds both R3-review files plus pyproject.toml, src/pams/__init__.py, and src/pams/types.py for 55 current non-self paths",
      "A005-R3-B1": "native pre-import embedded-CPython supervisor; exact installed-distribution, CPython-runtime, trusted-generated-source, wheel/ELF ledgers; project install/repository equality; complete trace-only origin validation",
      "A005-R3-B2": "NATP exact14 with one ordered K6 upstream; k1.upstream ordinals 0=G0 and 1=G1; exact evidence equality; 54-owner/106-edge acyclic simulation",
      "A005-R4-B1": "exact PyPreConfig/PyConfig 3.12.13 constructor/default/final maps, setter and call order, full readbacks/error/clear, and exact FD entry/locked/EOF/extra closure",
      "A005-R4-B2": "independent no-project bootstrap-only dataclasses capsule and zero-process precommit, two byte-identical outputs/root, then distinct final P06 full dispatch",
      "A005-R4-B3": "sole canonical association and event-specific origin preimages with exact normalization, discriminants, all wheel/repository/native/generated identities, equality and uniqueness",
      "A005-R5-B1": "writer-open sealed memfd handoff through /proc/self/fd, independent O_RDONLY OFD proof, writer close/EBADF, postclose reader verification, exact dup3 child installation, construction readback/error closure",
      "A005-R5-B2": "exact two-row ordered discovery evidence index/root with fresh process identity, launch/config/three FD readbacks, five-frame FD3 stream, equal output and joined attestation",
      "A005-R5-B3": "trusted set is exactly the common nonempty discovery rows and every full runtime consumes ordinals one-for-one through adjacent dedicated compile-to-exec pairs",
      "A005-R5-B4": "standalone exact canonical eleven-key trusted-row-plus-LF preimage plus sole class-to-kind and generator-digest mapping and complete trusted origin projection"
    },
    "negatives": [
      "normalized SmoothL1 replaced by unnormalized Huber, wrong e==delta branch, altered cast/operation/Neumaier order, FMA/reassociation, -0.0, or traversal_bounds supplied as run_bounds",
      "tune run_bounds not exact C-order <i4[[0,N_v]], stored traversal bounds not geometry-rederived, G1/evidence teacher-run scalar/wrong length/wrong seed order",
      "FE mode not FE_TONEAREST, MXCSR not 0x00001f80, FTZ/DAZ enabled, Torch flush-denormal enabled, denormal probe mismatch, MKLDNN/NNPACK/backend drift, unsupported readback, or reused checkpoint replay process",
      "K1 row hash excluding LF, hashing embedded parent/tag/ordinal, omitting null, pretty/CRLF encoding, row substitution, or Cdev digest not equal exact committed K1 row",
      "missing, extra, duplicate, substituted, or tampered winning or losing checkpoint artifact/receipt/output/evaluation receipt; score swap or incomplete 120 grid",
      "prediction component caller input, component relabel, feature swap, K1 omission/post-K1 subset, target swap, receipt-root self/extra/missing, vault preconsume access/retry, or stateful Mapping TOCTOU",
      "native supervisor path/argv/fd/PyConfig/search-path mismatch; python -m/shell/PATH/editable fallback; pams import before native hook; launcher/initial ELF omission; installed wheel/member or CPython stdlib/core ledger missing/extra/path/hash/member-map mismatch",
      "installed pams byte differs from repository row, pams __init__/types absent, purpose escalation from installation-equality/verifier-test to runtime, stdlib source treated as wheel/repo, pyc/zip/user-site/.pth load, or nonindexed builtin/frozen/extension",
      "trusted generated source row missing/extra/reordered, generated_source_hex/hash mismatch, wrong generator origin/qualname/filename/mode, unlisted anonymous eval/exec, generated exec after DISPATCH_BEGIN, or post-run trace mutating any precommitted authority index",
      "NATP uses 13 keys, omits/duplicates/reorders upstream_receipt_sha256, its sole element differs from k6_receipt_sha256/K6-RESAMPLER, or inventory/payload arrays differ",
      "K1 omits/swaps/duplicates k1.upstream ordinals, uses stage.upstream, substitutes G0/G1, mismatches G0 P402 manifest or G1 selection/run evidence, adds a catch-all edge, or 54-owner DAG does not produce exactly 106 acyclic roster-direct edges",
      "wrong PyPreConfig/PyConfig constructor/default/final field, omitted use_frozen_modules/encoding/path/orig_argv/private field, implicit preinitialization, PyConfig_Read, wrong setter order, incomplete runtime readback, missing/double PyConfig_Clear, or non-fail-closed PyStatus",
      "FD0/1/2/3/4/5/6 object/access/raw flag/inheritability/offset/size/seal/EOF/device/pipe mismatch, writable input, shared description, early offset drift, missing post-init readback, or any extra descriptor",
      "discovery capsule depends on final environment/code/trusted index, uses project/site-packages import, lacks durable zero-process precommit, differs across two outputs/roots, learns rows from full runtime, or authorizes its own process",
      "wheel member path normalization mismatch, null/present discriminant swap, missing wheel member/repository hash, arbitrary generated token, association alternate encoding/omitted null, duplicate association, or installed/wheel/repository equality failure",
      "free-form/digest-only trace origin, wrong event-kind schema, missing/extra identity key, alternate origin encoding, association/origin digest swap, path normalization alias, native build-id mismatch, generated row mismatch, or non-gap-free duplicate event order",
      "memfd writer closed before /proc reopen, proc open after EBADF, reader duplicated from writer instead of reopened, shared offset, wrong inode/seals/flags/EOF, writer surviving exec, child fd writable/CLOEXEC, missing/reordered construction step/readback, or alternate error stage",
      "discovery evidence has not exactly two run_id 0/1 rows, reuses process/pipe, shares/multiplexes FD3, uses unframed or alternate-endian stream, omits launch/config/one FD stage/output/attestation bytes, accepts truncation/trailing frame, unjoined/nonzero child, swaps readbacks, or binds output digest without exact evidence index/root",
      "trusted generated set empty, differs from either discovery, admits only a subset, adds a row, consumes rows out of order/more/less than once, accepts generic PY_CODE_EVAL/PY_CODE_EXEC for trusted origin, lacks adjacent compile-to-exec, or admits post-DISPATCH generation",
      "trusted_row_sha256 excludes LF/generated_source_hex, hashes parent/root/ordinal/tag, uses alternate JSON, resolves multiple/no rows, class/kind mapping differs from cpython-stdlib-source->cpython-stdlib, generator digest is renamed/rehashed/substituted, or origin projection fails any one-to-one field"
    ],
    "positives": [
      "independent binary64 boundary vectors on both sides and exact equality at delta reproduce bit-identical loss/reduction/objective bytes",
      "all 120 checkpoints in fresh locked subprocesses execute 56 one-view forwards with one-run certificate bounds and reproduce output/evaluation/score bytes; K7 independently repeats global selection once",
      "each Cdev K1 row foreign key equals SHA-256 of its exact standalone 13-key canonical row plus LF",
      "direct MappingProxy builder-to-consumer composition yields canonical bytes equal a plain dict while a state-changing mapping fails",
      "exact native supervisor is prevalidated and is the first authority image; native hook precedes Py_InitializeFromConfig and every pams/stdlib/wheel import; PROCESS_IMAGE/PREINIT rows plus complete joined trace validate only against already committed ledgers",
      "installed-distribution index covers every frozen wheel/install member; CPython index covers complete stdlib source/extension and executed builtin/frozen providers; every installed pams runtime byte is bijective and byte-equal to its purpose=runtime repository row",
      "two fresh P06 bootstrap-only no-project dataclasses discoveries under a prior capsule/precommit yield byte-identical outputs/root; a distinct full P06 consumes the derived trusted rows before DISPATCH_BEGIN and no generated event follows",
      "exact 54-owner inventory yields 106 direct roster edges with legal roles: NATP exact14 has one natp.k6 edge, K1 has two ordered k1.upstream edges, G1/run/stage edges retain prior exact ordinals, and topological sort consumes all nodes",
      "all 70 bound current paths rehash exactly: 24 TempoRAC sources, 19 tests, 23 planning/review files, and four runtime-bootstrap files, with fixed/timestamped amendment outputs excluded from self-binding",
      "exact CPython 3.12.13 release constructor maps contain all 10 Linux PyPreConfig and 64 Linux PyConfig fields; both mode profiles survive native/runtime/sys-stream readback, every setter/status/clear edge is observed, and FD mock entry/lock/post-init states match",
      "a precommitted no-project dataclasses capsule launched twice produces identical canonical output bytes/root; only afterward the trusted index/environment are built and a distinct final P06 dispatch consumes the exact rows",
      "present and generated installed-member fixtures reconstruct byte-identical association preimages, and every trace token reconstructs one event-specific origin object with exact wheel/repository/native/generated equality",
      "every sealed input completes all eight construction-readback stages while writer remains open through reader reopen, proves independent offsets and common inode, closes writer, verifies reader, and installs exact read-only child fd",
      "two sequential fresh discovery processes each yield one valid five-frame FD3 stream and joined attestation; the exact two-row evidence index validates all equality/difference rules and its digest/root bind the trusted index and final environment",
      "the common discovery output is nonempty and the full bootstrap emits exactly one adjacent dedicated compile-to-exec pair for each ordinal 0..R-1 with no other generated event",
      "each trusted-generated origin resolves one exact standalone eleven-key row+LF digest and the sole class/kind and generator-digest projections round-trip byte-identically"
    ],
    "round5_additions": [
      "CPython constructor-default tamper for any one of 10 PyPreConfig or 64 PyConfig fields; skipped/reordered setter, internal-read drift, absent PyConfig_Clear, wrong PyStatus branch, wrong stderr errors, or config/FD readback event mismatch fails.",
      "FD tamper for object class, access/status/descriptor flags, inheritability, offset, seekability, size, seal order/set, EOF token, device number, pipe capacity/queue, shared stdout/stderr description, retained fd3/4/5, reused closed number, or extra descriptor at any of three stages fails.",
      "Discovery capsule tamper for dataclasses bytes, initial native image, supervisor, argv/env/cwd/path/init/FD profile; truncated/filtered output, caller dataclass decoration, project import, final code/environment dependency, absent zero-process precommit, unequal twin output, or full-run reuse fails.",
      "Wheel raw ASCII member without UTF-8 bit is accepted only by the exact ASCII branch; invalid legacy byte, alternative decode, member/repository tag swap, normalized/raw mismatch, generated collision, association key rename/omitted null/alternate encoding fails.",
      "Every trace event kind round-trips its exact schema; origin key substitution, installed/member/repository mismatch, process launch-instance swap, FD/config readback digest swap, or event sha256 projection mismatch fails.",
      "Installer-generated row or wrapper/RECORD rewrite, CPython/supervisor misclassified as a wheel member, build-id tag/null mismatch, or native installed-association tag/null mismatch fails; exact empty generated-member index and byte-preserving wheel mapping pass.",
      "Required source-import/compile/exec and trusted compile/exec chains reuse the same canonical origin digest and pass; altered reuse, reuse outside a required chain, or distinct origin bytes under one digest fails."
    ],
    "round6_additions": [
      "Memfd construction mock records exactly eight stages and fourteen failure tokens; closing writer before proc reopen, opening reader by dup, shared-offset proof, wrong inode/seals/status/descriptor flags, surviving writer, nonzero child FD flags, missing transient close or alternate operation order fails.",
      "Valid memfd mock keeps writer open, seals it, opens /proc/self/fd/<writer> O_RDONLY, promotes the independent reader, proves offset independence, closes/proves-EBADF writer, reverifies reader and dup3-installs the exact child fd.",
      "Discovery FD3 parser accepts only the exact magic/count and five type/length/hash/payload frames on each separate pipe; zero/oversize/short/extra/reordered/duplicate frame, concatenated JSON, wrong endian/digest/schema/stage or shared stream fails.",
      "Discovery evidence index has exactly rows run_id/process_ordinal 0/0 then 1/1, distinct fresh process identity, exact launch/config/three FD/output bytes, joined attestation and required equality projections; omission, reuse, cross-row swap or output-only evidence fails.",
      "Trusted index rows are nonempty and byte-identical to both discovery output row arrays. Empty, subset, superset, reorder, duplicate or final-runtime-learned row fails.",
      "Full runtime emits dedicated trusted compile then byte-identical trusted exec for every ordinal exactly once before DISPATCH_BEGIN; generic eval/exec alternate, unmatched pair, gap, replay or later generation fails.",
      "trusted_row_sha256 independently reconstructs exact standalone eleven-key canonical row bytes including generated_source_hex and LF; parent/root/tag/ordinal/pretty/CRLF/omitted-field alternatives fail.",
      "The sole cpython-stdlib-source to cpython-stdlib mapping and exact generator_sha256 to generator_origin_sha256 equality reconstruct every trusted origin field; rename, rehash, provider swap, wrong output root or collision fails."
    ],
    "status": "SPECIFIED_NOT_RUN; no test execution is authorized by this amendment"
  },
  "round6_closure_matrix": {
    "A005-R3-B2": {
      "closure": "UNCHANGED",
      "rule": "NATP exact14, K1 G0/G1 upstream, 54 owners, 106 roster-direct backward edges, K6 lineage, and G5a typed root remain unchanged"
    },
    "A005-R4-B1": {
      "closure": "EXACTLY_SPECIFIED",
      "rule": "exact source-bound CPython 3.12.13 isolated preconfig/config freezes all 10+64 Linux fields/defaults/finals/setters/internal-read/readbacks/errors/clear; exact three-stage FD evidence closes object/flags/inheritability/offset/size/seals/EOF/extras"
    },
    "A005-R4-B2": {
      "closure": "EXACTLY_SPECIFIED",
      "rule": "an independent bootstrap-only dataclasses capsule fixes CPython/native/init/FD/argv/env/path bytes but no project/final-code/environment; durable zero-process precommit precedes two byte-identical complete generated-event outputs/root, then trusted/environment and a distinct full P06 run"
    },
    "A005-R4-B3": {
      "closure": "EXACTLY_SPECIFIED",
      "rule": "wheel and repository association discriminants plus sole ASCII/UTF-8 path normalization feed one exact 16-key canonical JSON+LF preimage; every trace event uses a closed kind schema, repeated ledger identity, typed readback, and exact event-byte digest projection"
    },
    "A005-R5-B1": {
      "closure": "EXACTLY_SPECIFIED",
      "rule": "writer stays open through exact procfs O_RDONLY reopen; independent OFD/device/inode/offset/seal/hash/EOF checks precede writer close and postclose reader/dup3 child verification, with closed step-readback and error schemas"
    },
    "A005-R5-B2": {
      "closure": "EXACTLY_SPECIFIED",
      "rule": "exactly two sequential fresh discovery processes each produce an independent five-frame FD3 stream; the ordered evidence index commits launch/config/three FD/output bytes, root and joined attestation and is digest/root-bound by trusted index and final environment"
    },
    "A005-R5-B3": {
      "closure": "EXACTLY_SPECIFIED",
      "rule": "trusted rows equal the two discovery outputs exactly and are required nonempty; each full runtime consumes ordinals once in order through one adjacent dedicated compile-to-exec pair, with no generic or post-boundary alternative"
    },
    "A005-R5-B4": {
      "closure": "EXACTLY_SPECIFIED",
      "rule": "trusted_row_sha256 hashes the exact standalone eleven-key row plus LF; one fixed class-to-kind mapping and direct generator digest equality plus complete origin projection reconstruct one unique committed row"
    },
    "preserved_closed_semantics": {
      "closure": "UNCHANGED",
      "rule": "all teacher/tune/score/natural/K1/Cdev/target/prediction/capability/F4 science, 27 jobs, 93 hours, data, thresholds, G5a/target schemas, and claims remain unchanged"
    }
  },
  "round7_native_wave2_closure": {
    "audit_and_trace": {
      "argument_projections": {
        "compile": {
          "closed_keys": [
            "filename",
            "source_bytes",
            "source_hex",
            "source_sha256",
            "source_type"
          ],
          "rule": "source_type is exactly str-utf8 or bytes. A str is encoded once with strict UTF-8 and no BOM; bytes are used verbatim. source_bytes, lowercase source_hex and source_sha256 bind the same complete positive byte string. filename is a strict Unicode scalar string encoded by the same canonical JSON rule; NUL, surrogate or alternate encoding fails.",
          "schema": "temporac.audit-args.compile.v5"
        },
        "ctypes_dlopen": {
          "closed_keys": [
            "requested_path"
          ],
          "rule": "requested_path is an absolute strict-UTF-8 non-NUL path. Null/self handles, relative paths and path aliases fail. The later provider barrier must resolve exactly this path and inode to one precommitted ELF row.",
          "schema": "temporac.audit-args.ctypes-dlopen.v5"
        },
        "exec": {
          "closed_keys": [
            "code_authority_sha256",
            "code_filename",
            "code_firstlineno",
            "code_name",
            "code_qualname",
            "source_authority_sha256"
          ],
          "code_authority": "code_authority_sha256 hashes the exact standalone temporac.code-object-projection.v5 object defined by audit_and_trace.code_object_projection, including LF. No marshal, repr, pointer, hash(), pickle or address enters the projection.",
          "rule": "source_authority_sha256 is exactly the pending compile row source hash or the unique precommitted file-backed source hash. The five direct code fields equal the code projection. The pending state and code projection together identify one event; filename/name alone never authorize it.",
          "schema": "temporac.audit-args.exec.v5"
        },
        "import": {
          "closed_keys": [
            "filename_or_null",
            "form",
            "meta_path_provider_descriptors",
            "module_name",
            "path_hooks_provider_descriptors",
            "sys_path"
          ],
          "provider_descriptor": {
            "closed_keys": [
              "binding_kind",
              "defining_origin",
              "defining_origin_sha256",
              "implementation_code_authority_sha256_or_null",
              "implementation_kind",
              "implementation_module",
              "implementation_qualname",
              "provider_kind",
              "provider_module",
              "provider_qualname",
              "provider_state",
              "provider_state_sha256",
              "schema"
            ],
            "derivation": "For binding_kind=meta-path-provider, the provider is the exact sys.meta_path element and the selected implementation is getattr(provider,'find_spec'), resolving a bound method to its exact __func__. For binding_kind=path-hook-provider, the provider and selected implementation are the exact sys.path_hooks element itself. provider_kind is exactly module-bound-class,module-bound-function,exact-instance,or closure-function. provider_module/provider_qualname are strict Unicode scalar strings from the provider or exact provider type; module-bound values and exact-instance types must be reachable by identity through that module plus dotted qualname with no <locals>. A closure-function must have type exactly FunctionType and is identified by code authority plus complete state, never module lookup alone. implementation_kind is python-code or native-callable; python-code requires an exact function/code object and nonnull code-authority digest, while native-callable requires null and a defining origin of cpython-core,cpython-extension,installed-extension,or native-build-extension. implementation_module/qualname are exact and independently resolved. defining_origin is one complete allowed v5 origin and defining_origin_sha256 hashes it. provider_state is the complete state projection and provider_state_sha256 hashes its standalone canonical bytes plus LF. Any monkeypatch, same-origin different class, BuiltinImporter/FrozenImporter swap, closure-state swap, ambiguous lookup, descriptor protocol substitution, repr/id/address or runtime-added provider fails.",
            "schema": "temporac.import-provider-descriptor.v5"
          },
          "provider_descriptor_row_closed_keys": [
            "descriptor",
            "descriptor_sha256"
          ],
          "provider_descriptor_row_rule": "Each array element has exactly provider_descriptor_row_closed_keys. descriptor is one complete temporac.import-provider-descriptor.v5 object and descriptor_sha256 hashes its canonical standalone bytes including LF. Runtime derives the descriptor from the actual object and selected implementation, then requires byte-identical equality to the corresponding BASE pre-dispatch-import-plan descriptor row. Array position and duplicate byte-identical descriptors are preserved; no set conversion or origin-only projection is accepted.",
          "provider_state_projection": {
            "dict_entry_closed_keys": [
              "key",
              "value"
            ],
            "digest_rule": "provider_state_sha256 is SHA-256 of the complete canonical top-level state object including LF. The state has exactly closure,defaults,instance_dict,kwdefaults,schema,slots. Each of its first five value fields other than schema is exactly one tagged projection object; absent semantic state is the null variant, never omitted JSON null.",
            "limits": "Projection is pointer-cycle-free, maximum depth 128 and maximum 65536 total variant nodes. Repeated immutable values outside the active recursion stack are re-encoded; an active identity cycle, limit overflow or unsupported object fails.",
            "schema": "temporac.import-provider-state.v5",
            "state_closed_keys": [
              "closure",
              "defaults",
              "instance_dict",
              "kwdefaults",
              "schema",
              "slots"
            ],
            "variant_closed_keys": {
              "bool": [
                "tag",
                "value"
              ],
              "bytes": [
                "bytes",
                "hex",
                "tag"
              ],
              "callable-ref": [
                "defining_origin",
                "defining_origin_sha256",
                "module",
                "qualname",
                "tag"
              ],
              "complex-binary64": [
                "imag_bits_be_hex",
                "real_bits_be_hex",
                "tag"
              ],
              "dict": [
                "entries",
                "tag"
              ],
              "float-binary64": [
                "bits_be_hex",
                "tag"
              ],
              "frozenset": [
                "items",
                "tag"
              ],
              "int": [
                "decimal",
                "tag"
              ],
              "list": [
                "items",
                "tag"
              ],
              "null": [
                "tag"
              ],
              "str": [
                "tag",
                "value"
              ],
              "tuple": [
                "items",
                "tag"
              ]
            },
            "variant_rule": "Every value is exactly one object whose tag selects its exact variant_closed_keys. Primitive/int/binary64/bytes/str/tuple/frozenset encodings are identical to the corresponding code_object_projection variant rules; list preserves order. dict.entries is an array of exact two-key rows ordered by complete canonical key-variant bytes and keys are unique by exact Python equality plus identical projection; arbitrary hash iteration never orders it. callable-ref permits only a class or function reachable by identity from an exact module plus dotted qualname without <locals>; it embeds the complete allowed defining origin and its digest. closure preserves cell order and uses an explicit null variant for an empty cell; defaults is exact tuple-or-null, kwdefaults and instance_dict exact dict-or-null, and slots is a dict of every statically declared slot name to its exact value or null when absent. Classes/module-bound providers use null state fields unless exact defaults/closure apply. No code object, module, file, memoryview, mutable cycle, arbitrary instance, repr, pickle or pointer is a value."
          },
          "rule": "module_name and each present path are strict Unicode scalar strings. Derive form without inference from the five stock CPython arguments: search requires filename_or_null=null and sys_path,meta_path_provider_descriptors,path_hooks_provider_descriptors each be an exact ordered array; resolved-provider requires a present absolute normalized filename and all three collection fields exact JSON null. Any mixed null pattern or other form fails. In search form sys_path preserves every exact string in order; each actual meta_path/path_hooks element is independently projected through provider_descriptor/provider_state_projection and must byte-equal the same-position descriptor row precommitted in the BASE import-plan artifact. Resolved-provider form is permitted only for the stock second native-extension import audit occurrence and its filename must equal the later resolved extension spec/ELF path. These projected arguments alone assert neither success nor final provider.",
          "schema": "temporac.audit-args.import.v5"
        }
      },
      "audit_buffer": {
        "allocation": "Immediately before PySys_AddAuditHook and after the exact FP lock, call mmap once with length 67108864, PROT_READ|PROT_WRITE and MAP_PRIVATE|MAP_ANONYMOUS, fd=-1, offset=0; require success and a nonexecutable anonymous mapping. No resize, second mapping, file backing, executable permission or Python allocator is permitted. Allocation/protection/readback failure maps AUDIT_BUFFER_ALLOC. The complete buffer is munmap'd exactly once only after Py_FinalizeEx and terminal mapping-set capture; release failure maps AUDIT_BUFFER_RELEASE.",
        "limits": "Before 0x05, at most 65536 buffered payload entries and at most 67108864 projected wire bytes are permitted. Internal buffer bytes are exactly the concatenation of uint8 type || uint64_be(payload_length) || raw32(SHA256(payload)) || payload for type0x10,0x11,or0x13, with no frame-sequence field, pointer, padding, allocator header or other metadata. Each entry occupies 41+payload_length internal bytes but contributes exactly 49+payload_length projected wire bytes after the future uint64 sequence is inserted. Before append, checked integer addition must prove entry count, internal bytes and projected total remain within their respective 65536/67108864/67108864 bounds; equality is allowed only for a complete entry. Callback type projection, catalog selection, canonical encoding, append or bound failure maps AUDIT_BUFFER_CAPTURE regardless of the enclosing CPython call. Drain/ordering/write failure during the one post-0x05 flush maps AUDIT_BUFFER_FLUSH. These four stage tokens are disjoint and chronological; none may be relabeled as a later CPython/import/stream stage.",
        "ordering": "The native hook assigns each raw row its zero-based gap-free raw-audit sequence at callback entry and completes its typed projection and canonical payload bytes immediately. From hook installation through Py_InitializeFromConfig return, every raw 0x10 entry is appended to this buffer in callback order without a child-frame sequence; at the one post-initialize barrier the supervisor validates the complete expected initialization prefix, appends its sole precommitted 0x13 when it has a dynamic-load group, then appends all initialization normalized 0x11 payloads strictly by raw sequence. Native supervisor rows and the initial mapping set emit directly before hook installation and never use this buffer. Buffered entries are never sorted,coalesced or renumbered. After complete 0x05 configuration and its direct normalized config row, flush the complete initialization prefix in buffer order: assign the next gap-free child-frame sequence, insert uint64 sequence, emit the standard frame and advance once per entry. Reset logical length to zero without changing counters; no buffered entry survives. Only after that flush does the supervisor begin the explicit top-level import plan. For each such call, nested raw 0x10 rows are framed directly as callbacks occur; when the call returns and the complete expected group validates, frame exactly one grouped 0x13 when present and then frame every normalized 0x11 row strictly by raw sequence. Every primary cites the preceding shared delta and mirrors cite its group without digest. No post-flush payload is appended to this buffer, and no callback payload from initialization emits before 0x05.",
        "schema": "temporac.preconfig-audit-buffer-profile.v5"
      },
      "authority_digest_projection": {
        "CPYTHON_CONFIG_READBACK": {
          "runtime-readback": "readback_sha256"
        },
        "CTYPES_DLOPEN": {
          "ctypes-dynamic-load": "elf_sha256"
        },
        "DISPATCH_BEGIN": {
          "dispatch-boundary": "source_allowlist_sha256"
        },
        "NATIVE_FD_READBACK": {
          "runtime-readback": "readback_sha256"
        },
        "PREINIT_NATIVE_IMAGE": {
          "native-image": "elf_sha256",
          "process-image": "elf_sha256"
        },
        "PROCESS_EXEC": {
          "process-image": "elf_sha256"
        },
        "PROCESS_IMAGE": {
          "process-image": "elf_sha256"
        },
        "PY_BUILTIN_IMPORT": {
          "cpython-core": "provider_sha256"
        },
        "PY_CODE_EVAL": {},
        "PY_CODE_EXEC": {
          "cpython-core": "provider_sha256",
          "cpython-stdlib": "runtime_member_sha256",
          "installed-wheel": "installed_member_sha256",
          "project-installed": "installed_member_sha256"
        },
        "PY_EXTENSION_LOAD": {
          "cpython-extension": "elf_sha256",
          "installed-extension": "elf_sha256",
          "native-build-extension": "elf_sha256"
        },
        "PY_FROZEN_IMPORT": {
          "cpython-core": "provider_sha256"
        },
        "PY_IMPORT_SOURCE": {
          "cpython-stdlib": "runtime_member_sha256",
          "installed-wheel": "installed_member_sha256",
          "project-installed": "installed_member_sha256"
        },
        "PY_SOURCE_COMPILE": {
          "cpython-stdlib": "runtime_member_sha256",
          "installed-wheel": "installed_member_sha256",
          "project-installed": "installed_member_sha256"
        },
        "PY_TRUSTED_GENERATED_COMPILE": {
          "trusted-generated": "generated_source_sha256"
        },
        "PY_TRUSTED_GENERATED_EXEC": {
          "trusted-generated": "generated_source_sha256"
        }
      },
      "catalog": {
        "construction": "Before BASE-bundle commitment, P00 builds and independently reviews a complete finite catalog from the exact CPython 3.12.13 source, exact native sources, exact installed closure and exact pre-dispatch import plan. Runtime observation may neither add nor reclassify a catalog row.",
        "digest_rule": "audit_event_policy_catalog_sha256 is SHA-256 of the complete canonical catalog bytes including LF. The digest is external and absent from the catalog.",
        "row_closed_keys": [
          "action",
          "arg_projection_schema",
          "arity",
          "emitter",
          "event_name",
          "mapped_token_or_null",
          "stage_set"
        ],
        "rules": [
          "action is exactly map, ignored-nonexecutive, or reject; emitter is exactly cpython-audit or native-supervisor. action=map requires mapped_token_or_null be exactly one event_tokens value, while ignored-nonexecutive or reject requires null.",
          "Rows are ordered by UTF-8 event_name bytes, numeric arity, arg_projection_schema, emitter, then mapped_token_or_null with null before string. stage_set is a nonempty ASCII-sorted unique array and the complete seven-field row identity is unique.",
          "At the callback, event_name, positional arity, exact argument projection type and current stage select a nonempty finite candidate set. The frozen import/compile/exec safe-barrier state machine then independently resolves exactly one outcome token, ignored optional failure, or rejection; that completed outcome must select exactly one candidate row by action and mapped_token_or_null. Thus rows for delayed provider/source/trusted outcomes may share the callback tuple but never the completed outcome. Every other event resolves immediately to exactly one row. Unknown name, arity, type, stage or completed outcome, zero/multiple final candidates, duplicate row, wildcard, prefix match or runtime-learned row fails.",
          "Every ignored-nonexecutive event is explicitly enumerated. Ordinary artifact/data/receipt/index open or mmap is excluded from normalized executable events only when its exact catalog row says ignored-nonexecutive and its bytes never become executable.",
          "code.__new__, function.__new__, marshal or pickle code loading, sys.addaudithook after native installation, sys.settrace, sys.setprofile, subinterpreters, Python process spawning, executable anonymous memory and unexpected ctypes.dlsym or ctypes.call_function are reject rows."
        ],
        "schema": "temporac.audit-event-policy-catalog.v5",
        "top_level_closed_keys": [
          "contract_sha256",
          "rows",
          "schema"
        ]
      },
      "code_object_projection": {
        "closed_keys": [
          "argcount",
          "cellvars",
          "code_bytes",
          "code_hex",
          "consts",
          "exceptiontable_bytes",
          "exceptiontable_hex",
          "filename",
          "firstlineno",
          "flags",
          "freevars",
          "kwonlyargcount",
          "linetable_bytes",
          "linetable_hex",
          "name",
          "names",
          "nlocals",
          "posonlyargcount",
          "qualname",
          "schema",
          "stacksize",
          "varnames"
        ],
        "constant_variants": {
          "bool": {
            "closed_keys": [
              "tag",
              "value"
            ],
            "value": "value is exact JSON true or false"
          },
          "bytes": {
            "closed_keys": [
              "bytes",
              "hex",
              "tag"
            ],
            "value": "bytes is a nonnegative integer; hex is lowercase even-length and decodes to exactly that many bytes"
          },
          "code": {
            "closed_keys": [
              "projection",
              "projection_sha256",
              "tag"
            ],
            "value": "projection is one recursively valid code-object-projection.v5 object; projection_sha256 hashes its standalone canonical bytes plus LF; nesting is finite and pointer-cycle-free"
          },
          "complex-binary64": {
            "closed_keys": [
              "imag_bits_be_hex",
              "real_bits_be_hex",
              "tag"
            ],
            "value": "each bits field is exactly sixteen lowercase hex digits of the corresponding IEEE-754 binary64 storage bits in network byte order; every NaN payload, infinity and signed zero is preserved"
          },
          "ellipsis": {
            "closed_keys": [
              "tag"
            ],
            "value": "tag is ellipsis"
          },
          "float-binary64": {
            "closed_keys": [
              "bits_be_hex",
              "tag"
            ],
            "value": "bits_be_hex is exactly sixteen lowercase hex digits of IEEE-754 binary64 storage bits in network byte order; every NaN payload, infinity and signed zero is preserved"
          },
          "frozenset": {
            "closed_keys": [
              "items",
              "tag"
            ],
            "value": "items contains each recursively encoded element exactly once, ordered by its complete canonical variant bytes without terminal LF; duplicate encoded bytes or digest-only ordering fails"
          },
          "int": {
            "closed_keys": [
              "decimal",
              "tag"
            ],
            "value": "decimal is exactly 0 or -?[1-9][0-9]* and denotes the unbounded Python integer; plus, leading zero and negative zero are forbidden"
          },
          "not-implemented": {
            "closed_keys": [
              "tag"
            ],
            "value": "tag is not-implemented"
          },
          "null": {
            "closed_keys": [
              "tag"
            ],
            "value": "tag is null and represents Python None"
          },
          "str": {
            "closed_keys": [
              "tag",
              "value"
            ],
            "value": "value is the exact Python Unicode scalar sequence; lone surrogate and alternate normalization are rejected and canonical JSON encodes the unchanged string"
          },
          "tuple": {
            "closed_keys": [
              "items",
              "tag"
            ],
            "value": "items preserves exact tuple order and contains recursively encoded variants"
          }
        },
        "hash_rule": "code_authority_sha256 is SHA-256 of the complete canonical projection bytes including exactly one LF. For code_hex, linetable_hex and exceptiontable_hex, the paired byte count is nonnegative and lowercase hex decodes to exactly that many CPython bytes. argcount,posonlyargcount,kwonlyargcount,nlocals,stacksize,firstlineno are JSON integers in [0,2147483647]; flags is in [0,4294967295]. filename,name,qualname and every names/varnames/freevars/cellvars element are exact Unicode scalar strings; the four arrays preserve CPython tuple order. consts preserves co_consts order. Unsupported constant type, mutable value, recursive pointer cycle, missing/extra field, alternate endian or alternate tag fails.",
        "schema": "temporac.code-object-projection.v5",
        "tag_rule": "Each constant is exactly one object whose tag is the literal variant key above and whose keys equal that variant's closed_keys. No JSON primitive directly represents a constant, no variant omits its tag, and no extra value/count/digest field is permitted.",
        "variant_selection": "Select by exact CPython object identity/type, not isinstance: None, Ellipsis and NotImplemented use only their exact singletons; type(x) is bool,int,float,complex,bytes,str,tuple or frozenset selects the same-named variant; PyCode_CheckExact selects code. Bool is therefore never int. Subclasses, bytearray, memoryview, list, mutable container and every other type fail. Tuple/frozenset recursion applies this same rule at every element and detects repeated in-progress object identity before descent."
      },
      "dynamic_mapping_closure": {
        "delta": {
          "closed_keys": [
            "after_mapping_set_bytes",
            "after_mapping_set_bytes_hex",
            "after_mapping_set_sha256",
            "before_mapping_set_bytes",
            "before_mapping_set_bytes_hex",
            "before_mapping_set_sha256",
            "load_group_ordinal",
            "new_rows",
            "primary_elf_sha256s",
            "raw_parent_sequences",
            "schema"
          ],
          "digest_rule": "dynamic_mapping_delta_sha256 is SHA-256 of the complete canonical delta bytes including LF and is absent from the object. Each bytes value is positive; lowercase hex decodes to exactly the complete canonical mapping-set bytes including LF and the paired digest hashes them. load_group_ordinal is a nonnegative gap-free import-barrier group number. raw_parent_sequences is a nonempty strictly increasing integer array and primary_elf_sha256s is an equal-length array of exact ELF digests; their positions bind every successful primary PY_EXTENSION_LOAD or CTYPES_DLOPEN occurrence in the group.",
          "rule": "A group is exactly one of two supervisor-owned boundaries. For owner_kind=cpython-initialization, before is the already emitted initial-preinitialize mapping set and after is captured immediately after Py_InitializeFromConfig returns. For owner_kind=top-level-import, the supervisor captures before immediately before one exact fixed-plan PyImport_ImportModule call and after only when that call returns to the same C loop. This grouped safe barrier deliberately claims no nonexistent post-dlopen callback. The group's raw_parent_sequences/primary_elf_sha256s equal the exact future-precommitted expected_dynamic_load_groups row and include every nested extension/ctypes primary completed during that boundary; search mirrors are excluded. after contains every before row byte-for-byte with identical multiplicity, including kernel rows. new_rows is exact multiset difference in after order, only kind=file-elf, and may be empty only when every primary ELF was already in before. Every primary is in after; every new nonprimary row is reachable by a finite exact DT_NEEDED edge path from at least one listed primary. Shared dependency stays one collective row and is never guessed onto one caller. Removal, kernel change, identity/segment drift, raw/group mismatch, unrelated ELF, missing dependency, basename substitution, digest-only set or unmapped primary fails.",
          "schema": "temporac.dynamic-executable-mapping-delta.v5"
        },
        "extension_group_rule": "For one logical extension import, exactly resolved-provider occurrence is primary: primary raw sequence equals its raw reference, load-group ordinal identifies enclosing initialization or top-level-import barrier, and delta digest is present. Stock earlier search occurrence is mirror: same extension,later primary sequence,group ordinal,null delta. Both buffer until post-initialize or PyImport-return barrier, then normalize by raw sequence; mirror never enters primary arrays. Every extension/ctypes primary in same boundary repeats one collective digest and arrays list each once. Exactly one primary+earlier mirror exist when stock emits both; single primary only when accepted CPython path emits no search occurrence. Cross-module mirror,missing group,inconsistent digest or unlisted raw fails. No per-extension post-return interposition/dependency attribution is asserted.",
        "kernel_runtime_evidence": {
          "closed_keys": [
            "architecture",
            "contract_sha256",
            "kernel_build_id",
            "kernel_image_bytes",
            "kernel_image_sha256",
            "kernel_release",
            "schema",
            "vdso_bytes",
            "vdso_sha256",
            "vsyscall_mode"
          ],
          "digest_rule": "kernel_runtime_evidence_sha256 is SHA-256 of the complete canonical ten-key bytes including LF and is absent from the object.",
          "rule": "This future-populated P00 artifact is reviewed before BASE and binds the exact x86_64 Linux kernel image retrievable bytes/hash/build-id/release plus the exact in-memory [vdso] executable mapping bytes/hash for that runtime. architecture=x86_64; counts are positive; hashes are lowercase over complete bytes; vsyscall_mode is none,emulate,or native and must equal boot/kernel readback. It contains no launch/run/evidence output. The child rehashes [vdso] bytes at initial and terminal capture; [vsyscall] is not read when permissions deny access and is authorized only by the exact mode. Kernel/image/vdso mismatch, alternate boot, unreadable required VDSO or fabricated hash fails before child authority.",
          "schema": "temporac.kernel-runtime-evidence.v5"
        },
        "mapping_evidence": {
          "row_closed_keys": [
            "frame_sequence",
            "kind",
            "load_group_ordinal_or_null",
            "payload_bytes",
            "payload_bytes_hex",
            "payload_sha256",
            "stage"
          ],
          "row_rule": "Rows are exact child-frame order and frame_sequence is gap-free. kind=mapping-set requires type0x12, group ordinal null and stage initial-preinitialize or terminal-after-finalize; payload is exact mapping_set bytes. kind=mapping-delta requires type0x13, nonnegative gap-free group ordinal and dynamic-after; payload is exact grouped delta with same ordinal, and every listed primary origin repeats payload_sha256/group while mirrors repeat ordinal but null digest. Counts/hex/hash bind complete payload including LF. Initial is first,terminal last, and interior rows are bijective in order with complete expected_dynamic_load_groups: at most first is cpython-initialization, all remaining are top-level-import. No per-loader,mirror-owned or unframed delta exists.",
          "schema": "temporac.executable-mapping-evidence-index.v5",
          "top_level_closed_keys": [
            "process_ordinal",
            "rows",
            "schema"
          ]
        },
        "mapping_set": {
          "digest_rule": "executable_mapping_set_sha256 is SHA-256 of the complete canonical mapping-set bytes including LF and is absent from the object.",
          "file_row_closed_keys": [
            "absolute_path",
            "build_id_or_null",
            "build_id_tag",
            "device_major",
            "device_minor",
            "elf_sha256",
            "executable_segments",
            "inode",
            "kind",
            "mapping_ordinal",
            "provider_class",
            "provider_evidence_kind",
            "provider_evidence_sha256",
            "schema"
          ],
          "kernel_row_closed_keys": [
            "kernel_mapping_sha256_or_null",
            "kind",
            "mapped_bytes",
            "mapping_name",
            "permissions",
            "provider_evidence_sha256",
            "schema"
          ],
          "row_rule": "Each row is discriminated by kind. kind=file-elf requires exactly file_row_closed_keys: absolute_path is normalized from the exact /proc/self/maps pathname after rejecting deleted, memfd, bracket, relative, symlink-race or ambiguous paths; device/inode are exact nonnegative integers and byte-equal fstat; ELF bytes/build-id/DT_NEEDED/provider_class resolve one precommitted ELF row; provider_evidence_kind is cpython-runtime-member,installed-member,native-build-output,native-install-member,initial-native-image,or system-runtime-elf and its digest resolves that exact typed row, never a basename. kind=kernel-exec requires exactly kernel_row_closed_keys: mapping_name is [vdso] or [vsyscall], provider_evidence_sha256 resolves the exact BASE kernel-runtime-evidence; [vdso] has permissions=r-xp, positive mapped_bytes equal vdso_bytes and nonnull exact in-memory hash, while [vsyscall] has permissions=--xp, mapped_bytes exactly 4096, null hash and is present iff vsyscall_mode is emulate or native. No other bracket/anonymous executable mapping exists. File executable_segments are nonempty and match maps+dl_iterate; mapping_ordinal is zero-based among identical identities. Kernel rows occur first in mapping_name byte order, then file rows by UTF-8 absolute_path,device_major,device_minor,inode,mapping_ordinal; all rows are unique.",
          "schema": "temporac.executable-mapping-set.v5",
          "segment_closed_keys": [
            "file_offset_bytes",
            "mapped_bytes",
            "permissions"
          ],
          "segment_rule": "File segments are positive mapped_bytes, nonnegative page-aligned file_offset_bytes, exact permissions r-xp or --xp, and ordered by file offset then permissions then length. The set is captured by one `/proc/self/maps` open/read/EOF plus one dl_iterate_phdr traversal with exact path/inode/load-segment reconciliation; kernel rows are reconciled separately to kernel-runtime-evidence. Transient capture descriptors are opened only at the specified safe barrier, closed and proved EBADF. Address/ASLR values are verified for reconciliation but deliberately absent from canonical bytes.",
          "stage_values": [
            "initial-preinitialize",
            "dynamic-before",
            "dynamic-after",
            "terminal-after-finalize"
          ],
          "top_level_closed_keys": [
            "contract_sha256",
            "rows",
            "schema",
            "stage"
          ]
        },
        "terminal_rule": "The child emits exact 0x12 initial-preinitialize mapping-set bytes captured after PROCESS_IMAGE/PREINIT_NATIVE_IMAGE reconciliation and before Py_PreInitialize, one 0x13 for the optional precommitted CPython-initialization group and then one after each precommitted top-level import load group, then exact 0x12 terminal-after-finalize bytes immediately after successful Py_FinalizeEx. Initial and terminal contain the same byte-identical kernel-exec [vdso]/optional [vsyscall] rows. Apply grouped deltas in gap-free load_group_ordinal order to the initial row multiset; each before rows projection must equal current after changing only stage to dynamic-before, and the resulting after rows become current. The final current rows, after changing only stage to terminal-after-finalize and recanonicalizing, byte-equal the independently recaptured terminal set. Mirrors change nothing. DISPATCH_BEGIN is permitted only after the last planned group and thereafter any raw extension/ctypes event or executable-set change rejects. This proves every automatic DT_NEEDED/transitive mapping and zero unaccounted executable mapping without requiring stock CPython to expose post-dlopen return or creating a seventeenth normalized token. Any other anonymous/bracket PROT_EXEC, JIT/code mmap, kernel-row change, dlclose/removal or executable set change outside one group fails."
      },
      "event_order": {
        "controller_reserved_sequence": "For each child process_ordinal, the controller reserves normalized sequence 0 from the exact fork launch/request but does not claim exec success from fork return or an undefined handshake. Only after it validates the child's unique PROCESS_IMAGE row, exact supervisor ELF/AT_EXECFN identity, terminal EOF and waitpid exit zero does it construct exactly one PROCESS_EXEC row and prepend it at reserved sequence 0 during final evidence assembly. This row is absent from the child stream; execve failure or missing/mismatched child identity creates no PASS row.",
        "discovery_boundary": "Each discovery child has no DISPATCH_BEGIN. After all pre-dispatch imports, the complete trusted-candidate prefix and config/FD/native evidence, it emits the unique discovery output, executes Py_FinalizeEx while recording every remaining raw catalog event and any mapped normalized event, then emits terminal success and EOF. Generated or other executable-origin events during finalization fail; cataloged ignored-nonexecutive shutdown events remain exact raw evidence. Generated rows remain evidence-only.",
        "full_boundary": "The full child consumes the complete trusted sequence, then emits exactly one DISPATCH_BEGIN before dereferencing the invocation input index or calling Python dispatch. Every executable/import/native origin needed by every handler must already be loaded by the complete pre-dispatch plan. After this boundary generated compile/exec, import, source compile/exec audit, ctypes load, process spawn and any new executable mapping each have exact multiplicity zero, including during Py_FinalizeEx; any occurrence fails. Cataloged ignored-nonexecutive shutdown events are still emitted as raw rows between the unique dispatch-result frame and terminal success.",
        "per_process_sequence": "Child normalized sequence begins at 1 because sequence 0 is reserved. The completed carrier-validated-and-closed NATIVE_FD_READBACK is exactly sequence 1 because launch identity is unavailable before that transport boundary. PROCESS_IMAGE follows exactly, then PREINIT_NATIVE_IMAGE rows appear in initial-native-image-index order after a closed temporary maps/exe read and before Py_PreInitialize. Post-carrier-exec-entry and locked-pre-cpython NATIVE_FD_READBACK rows follow in that order; post-pyinitialize NATIVE_FD_READBACK and CPYTHON_CONFIG_READBACK follow initialization; raw-import/compile/exec/dlopen normalized events then follow actual safe-barrier completion order. The final controller-assembled array is ordered by process_ordinal then sequence, unique and gap-free. Raw audit sequence is an independent zero-based gap-free counter referenced by raw_event_sequence_or_null.",
        "serialization_rule": "Carrier/process/preinit/native-FD rows and the initial mapping set are emitted directly in the exact early success order before hook installation. Only callback-derived raw/mapping/normalized payloads occurring before the fixed configuration frame follow audit_buffer: raw/normalized semantic sequence is fixed at event/completion time, but child-frame sequence is assigned at the one post-0x05 flush. After that flush rows are framed directly. Committed semantic sequence remains actual completed-evidence order independent of this explicitly deferred wire emission. Every transient fd used to read process maps/exe is opened only after the physical FD7 scan/close, closed and proved EBADF before the next operational readback. The controller verifies exact reserved-row plus child-row continuity and never renumbers or rewrites child bytes. Missing, extra, duplicate, gap, overflow, reordered semantic capture, conflicting raw reference or child-supplied PROCESS_EXEC fails."
      },
      "event_semantics": {
        "CPYTHON_CONFIG_READBACK": "Exactly one runtime-readback origin after successful Py_InitializeFromConfig and post-pyinitialize FD readback; it binds complete initialization profile, launch, config/global/sys/stream bytes and hash.",
        "CTYPES_DLOPEN": "One ctypes-dynamic-load origin for each successful permitted ctypes.dlopen, normalized only at its safe barrier after exact requested primary path and complete before/after executable-mapping delta resolution. The delta contains the primary plus every newly mapped DT_NEEDED/transitive ELF and no unrelated mapping. Failed, empty, uncommitted or partially mapped loads reject.",
        "DISPATCH_BEGIN": "Exactly one in full-runtime and zero in discovery; dispatch-boundary origin binds exact owner and source allowlist. It occurs after all required imports/trusted rows and before input-index dereference.",
        "NATIVE_FD_READBACK": "Exactly four per child in carrier-validated-and-closed, post-carrier-exec-entry, locked-pre-cpython, post-pyinitialize order. Each runtime-readback origin binds the complete carrier or operational readback bytes/hash, launch and applicable transport/fd profile.",
        "PREINIT_NATIVE_IMAGE": "Exactly one row for every row of the precommitted initial-native-image-index, in that index order, using process-image for the supervisor image and native-image for every other mapped ELF. The captured map precedes Py_PreInitialize.",
        "PROCESS_EXEC": "Exactly one controller-created sequence-zero process-image row per successful child, binding controller request, exact supervisor ELF and that child launch. It is finalized only after the matching child PROCESS_IMAGE plus EOF/waitpid PASS prove exec and is then prepended during controller assembly; no child emits it and no exec failure receives it.",
        "PROCESS_IMAGE": "Exactly one child-created process-image row immediately after the completed carrier-readback row, binding /proc/self/exe, AT_EXECFN, installed supervisor/ELF, controller request and launch byte-for-byte.",
        "PY_BUILTIN_IMPORT": "cpython-core origin with provider_kind=builtin and safe-barrier-resolved module/spec equality.",
        "PY_CODE_EVAL": "Exact multiplicity zero. Stock CPython exec audit cannot distinguish eval and no event may be guessed or relabeled as this token.",
        "PY_CODE_EXEC": "File-backed cpython-core/cpython-stdlib/project-installed/installed-wheel origin matching the exact code-object and source authority. Trusted generated exec is forbidden here and uses its dedicated token.",
        "PY_EXTENSION_LOAD": "cpython-extension,installed-extension or native-build-extension origin resolved at its initialization/import group barrier through exact runtime/association/native-install/ELF membership; _fp_control is only native-build-extension. Exactly one primary occurrence per logical extension repeats the complete shared group-delta digest and appears in its aligned primary array; a stock search mirror embeds no digest and points to that primary. Dependencies belong collectively to the group, not one extension.",
        "PY_FROZEN_IMPORT": "cpython-core origin with provider_kind=frozen and exact frozen/runtime ledger identity.",
        "PY_IMPORT_SOURCE": "cpython-stdlib, project-installed or installed-wheel origin with complete provider/source bytes and ledger equality, emitted only after successful import barrier.",
        "PY_SOURCE_COMPILE": "Same byte-identical file-backed origin as the unique pending source authority and exact raw compile projection. There is no compile-mode field.",
        "PY_TRUSTED_GENERATED_COMPILE": "Trusted-generated origin reconstructing the next exact discovery/trusted row. Discovery labeling is non-authoritative; full runtime must match the precommitted row before execution.",
        "PY_TRUSTED_GENERATED_EXEC": "Same byte-identical trusted-generated origin as the immediately preceding dedicated compile; it consumes that row exactly once. No delayed, reused or post-boundary event exists."
      },
      "event_tokens": [
        "CPYTHON_CONFIG_READBACK",
        "CTYPES_DLOPEN",
        "DISPATCH_BEGIN",
        "NATIVE_FD_READBACK",
        "PREINIT_NATIVE_IMAGE",
        "PROCESS_EXEC",
        "PROCESS_IMAGE",
        "PY_BUILTIN_IMPORT",
        "PY_CODE_EVAL",
        "PY_CODE_EXEC",
        "PY_EXTENSION_LOAD",
        "PY_FROZEN_IMPORT",
        "PY_IMPORT_SOURCE",
        "PY_SOURCE_COMPILE",
        "PY_TRUSTED_GENERATED_COMPILE",
        "PY_TRUSTED_GENERATED_EXEC"
      ],
      "filter": "Normalize only the 16 preserved executable/runtime tokens. Ordinary artifact, data, JSON, NPZ, NPY, receipt, index, log and result open/read/stat/enumerate/mmap events remain raw catalog evidence only and are ignored-nonexecutive unless those same bytes subsequently become executable input, in which case the missing executable transition rejects.",
      "final_trace": {
        "assembly": "The controller, only after terminal child frame, EOF and waitpid, joins child raw/normalized streams with process evidence. A child never records its own join; a join is not a seventeenth normalized event token.",
        "child_stream_index": {
          "row_closed_keys": [
            "bytes",
            "bytes_hex",
            "eof_observed",
            "launch_instance_sha256",
            "process_ordinal",
            "sha256",
            "terminal_frame_sequence"
          ],
          "rule": "Rows are exact process-ordinal order, one for full-runtime and two for discover-twin. bytes_hex reconstructs the complete magic-through-last-frame stream excluding EOF, sha256 hashes it, eof_observed=true, terminal_frame_sequence names the unique final 0x7e/0x7f sequence, and no child stream contains its wait result.",
          "schema": "temporac.native-child-stream-index.v5",
          "top_level_closed_keys": [
            "mode",
            "rows",
            "schema"
          ]
        },
        "closed_keys": [
          "child_stream_index_sha256",
          "contract_sha256",
          "controller_request_sha256",
          "executable_mapping_evidence",
          "initial_executable_mapping_set_sha256",
          "launch_evidence_index_sha256",
          "launch_instance_sha256",
          "normalized_events",
          "process_joins",
          "raw_audit_events",
          "schema",
          "status",
          "terminal_executable_mapping_set_sha256"
        ],
        "join_row_closed_keys": [
          "child_exit_code_or_null",
          "child_pid",
          "child_signal_or_null",
          "child_starttime_ticks",
          "child_stream_eof",
          "mode",
          "process_ordinal",
          "wait_status_u32"
        ],
        "mapping_evidence_rule": "executable_mapping_evidence is the exact parsed temporac.executable-mapping-evidence-index.v5 projection of every 0x12/0x13 frame in child order. First/last are sole initial/terminal sets with matching trace digests. Interior deltas are gap-free and bijective with complete expected_dynamic_load_groups: at most first owner is cpython-initialization and all remaining owners are top-level-import. Each raw_parent_sequences/primary_elf_sha256s equals its precommitted group; every primary repeats group ordinal/digest, mirrors only ordinal/primary reference. No group is learned, no primary belongs to two groups, no unlisted primary exists. Ordered replay yields terminal bytes/digest equal child terminal field.",
        "process_join_index": {
          "rule": "The exact standalone index has keys child_stream_index_sha256,mode,rows,schema. rows use join_row_closed_keys in process ordinal order and are constructed only after draining each named stream to EOF and waitpid on that exact positive pid; child_starttime_ticks is the positive /proc identity captured at fork and must equal discovery/full attestation evidence, preventing PID reuse. process_join_index_sha256 hashes complete canonical index bytes including LF; no join is emitted by a child or normalized as an event.",
          "schema": "temporac.native-process-join-index.v5"
        },
        "schema": "temporac.executable-origin-trace.v5",
        "status": "PASS_ONLY_AFTER_ALL_ORIGINS_RESOLVE",
        "trace_hash_rule": "final_trace_sha256 is SHA-256 of the complete thirteen-key canonical trace bytes including LF. launch_evidence_index_sha256 resolves the exact one-row full-runtime launch evidence and its launch hash equals launch_instance_sha256. raw_audit_events are exact gap-free child-frame rows. normalized_events are, for each process, the controller-created PROCESS_EXEC sequence-zero row followed byte-for-byte by the child-frame rows beginning at sequence one; the merged array is process-ordinal/sequence order with no gap or rewrite. executable mapping evidence obeys mapping_evidence_rule. process_joins is exact process-ordinal order and byte-equals the separately hashed join index rows. The trace has no own hash."
      },
      "mapping": [
        {
          "arity": 5,
          "audit_event": "import",
          "rule": "Project the five stock arguments into the exact search or resolved-provider discriminated form without repr or pointer identity. Before Py_InitializeFromConfig returns, each raw occurrence belongs to the closed expected_initialization_raw_rows prefix and remains pending until the one post-initialize safe barrier; afterward it belongs to exactly one supervisor-owned top-level PyImport call and remains pending until that call returns. At either boundary validate the complete precommitted raw/normalized projection, capture the one optional grouped mapping delta spanning that boundary, then consume/append normalized rows strictly by ascending same-process raw sequence; dependency, module, provider, path or completion sorting is forbidden. A successful sys.modules[name].__spec__ provider, normalized origin and loader resolves through exactly one BASE/FINAL ledger row, then emits exactly one PY_BUILTIN_IMPORT,PY_FROZEN_IMPORT,PY_IMPORT_SOURCE or PY_EXTENSION_LOAD. A resolved-provider occurrence may emit only PY_EXTENSION_LOAD and its filename equals that origin; search may emit any of four. For a logical extension pair, resolved-provider is primary and earlier search is mirror; both reference the same precommitted load group, only primary enters group arrays and repeats shared nonnull digest. All primaries in one initialization/import boundary share one group delta, collectively proving dependencies without per-callback post-return inference. Initialization failures are never optional; later failed imports emit no normalized row only by the next exact optional_absence row. Any occurrence outside these boundaries, failed/ambiguous/duplicate-unresolved/pre-resolution classification or post-DISPATCH_BEGIN import fails."
        },
        {
          "arity": 2,
          "audit_event": "compile",
          "rule": "Arguments are source and filename only; CPython supplies no compile mode and no compile_mode may be inferred. A match to one precommitted file-backed source row emits PY_SOURCE_COMPILE. In discovery only, a non-file-backed compile with one exact source-backed Python generator frame enters the pending candidate state and is normalized as PY_TRUSTED_GENERATED_COMPILE only after its adjacent exec completes the output row; this label grants no authority. In full runtime, the same event must byte-match the next already committed trusted row before emitting PY_TRUSTED_GENERATED_COMPILE. Other compile fails."
        },
        {
          "arity": 1,
          "audit_event": "exec",
          "rule": "Project the code object through its exact verified source or pending generated identity without repr, address or marshal serialization. In discovery, exactly the adjacent exec closes one pending candidate row and is then normalized as PY_TRUSTED_GENERATED_EXEC without granting authority. In full runtime, a matching pending already-committed trusted row emits PY_TRUSTED_GENERATED_EXEC; every other allowed source-backed execution emits PY_CODE_EXEC. Stock exec cannot distinguish eval, so PY_CODE_EVAL has exact multiplicity zero and is never guessed."
        },
        {
          "arity": 1,
          "audit_event": "ctypes.dlopen",
          "rule": "Record the typed requested path and raw sequence as a pending primary inside either the CPython-initialization prefix or the currently active supervisor-owned top-level import call; the stock audit event occurs before dlopen and no post-return callback is assumed. At the post-initialize barrier or when that outer PyImport call returns, resolve the exact requested ELF and consume this row at the one grouped mapping barrier. Its ctypes-dynamic-load origin repeats the precommitted load-group ordinal/shared delta digest and its own raw sequence; group arrays contain it exactly once with every extension/ctypes primary in that boundary. A ctypes.dlopen outside initialization/pre-dispatch groups, after DISPATCH_BEGIN, missing from expected_dynamic_load_groups, failed/ambiguous provider, or executable change outside the group fails."
        }
      ],
      "native_tokens": {
        "emitted_only_by_supervisor": [
          "CPYTHON_CONFIG_READBACK",
          "DISPATCH_BEGIN",
          "NATIVE_FD_READBACK",
          "PREINIT_NATIVE_IMAGE",
          "PROCESS_EXEC",
          "PROCESS_IMAGE"
        ],
        "python_hook_cannot_forge": true
      },
      "normalized_row_closed_keys": [
        "event",
        "origin",
        "origin_sha256",
        "process_ordinal",
        "raw_event_sequence_or_null",
        "sequence",
        "sha256"
      ],
      "normalized_row_hash_rule": "sha256 hashes the standalone canonical six-key projection event,origin,origin_sha256,process_ordinal,raw_event_sequence_or_null,sequence plus LF. The seven-key stored row is then reconstructed by adding sha256. origin itself recanonicalizes and hashes to origin_sha256; neither digest is self-referential.",
      "normalized_row_rule": "event is exactly one event_tokens value and origin kind is allowed by origin_union.event_allowed_kinds. PROCESS_EXEC,PROCESS_IMAGE,PREINIT_NATIVE_IMAGE,NATIVE_FD_READBACK,CPYTHON_CONFIG_READBACK and DISPATCH_BEGIN are native-supervisor emissions with raw_event_sequence_or_null=null; all PY_* and CTYPES_DLOPEN normalized rows have a nonnegative raw reference resolving exactly one same-process raw row whose catalog mapping and argument projection produce this event/origin. process_ordinal and sequence obey event_order. authority_digest_projection[event][origin.kind] names the origin field whose underlying exact bytes are independently rehashed and compared; it is not the normalized row digest. sha256 only obeys normalized_row_hash_rule. A missing/extra raw link, wrong authority projection, native token from Python, or two normalized rows from one raw row fails.",
      "normalized_row_schema": "temporac.normalized-executable-event.v5 (external structural schema; the seven-key row intentionally has no schema member)",
      "origin_union": {
        "digest_rule": "origin_sha256 is SHA-256 of the exact standalone kind-specific origin object serialized as canonical compact sorted-key UTF-8 JSON with one LF. The normalized row embeds that parsed object and repeats the digest. Alternate tagged-H/binary/string packing, delimiter concatenation, omitted null, renamed key, v4 schema token, path alias or digest-only origin fails.",
        "event_allowed_kinds": {
          "CPYTHON_CONFIG_READBACK": [
            "runtime-readback"
          ],
          "CTYPES_DLOPEN": [
            "ctypes-dynamic-load"
          ],
          "DISPATCH_BEGIN": [
            "dispatch-boundary"
          ],
          "NATIVE_FD_READBACK": [
            "runtime-readback"
          ],
          "PREINIT_NATIVE_IMAGE": [
            "native-image",
            "process-image"
          ],
          "PROCESS_EXEC": [
            "process-image"
          ],
          "PROCESS_IMAGE": [
            "process-image"
          ],
          "PY_BUILTIN_IMPORT": [
            "cpython-core"
          ],
          "PY_CODE_EVAL": [],
          "PY_CODE_EXEC": [
            "cpython-core",
            "cpython-stdlib",
            "installed-wheel",
            "project-installed"
          ],
          "PY_EXTENSION_LOAD": [
            "cpython-extension",
            "installed-extension",
            "native-build-extension"
          ],
          "PY_FROZEN_IMPORT": [
            "cpython-core"
          ],
          "PY_IMPORT_SOURCE": [
            "cpython-stdlib",
            "installed-wheel",
            "project-installed"
          ],
          "PY_SOURCE_COMPILE": [
            "cpython-stdlib",
            "installed-wheel",
            "project-installed"
          ],
          "PY_TRUSTED_GENERATED_COMPILE": [
            "trusted-generated"
          ],
          "PY_TRUSTED_GENERATED_EXEC": [
            "trusted-generated"
          ]
        },
        "field_equality": {
          "cpython providers": "Every module/provider/runtime path and member hash equals one exact CPython runtime, builtin, frozen, source or extension ledger row; provider_kind is exactly builtin or frozen for cpython-core.",
          "dynamic loads": "A ctypes-dynamic-load or primary extension origin's dynamic_mapping_delta_sha256 resolves exactly one type0x13 grouped mapping-evidence row with the same dynamic_mapping_load_group_ordinal. The origin's raw reference and primary ELF occur at the same position in that delta's raw_parent_sequences/primary_elf_sha256s arrays. Every primary in one group repeats the same digest; this intentional many-to-one reference does not duplicate the frame. Extension role is primary or mirror; primary has a nonnull digest and primary raw sequence equal its normalized raw reference, while mirror has null digest and points to the exact same-module primary under extension_group_rule. ctypes-dynamic-load always has one nonnull shared group digest and raw_parent_sequence equals its normalized raw reference. Every automatic dependency is represented collectively inside the group delta and terminal mapping equality; it is never silently attributed to one primary ELF.",
          "installed providers": "association_sha256 resolves one exact sixteen-key installed-member association. distribution,version,installed_path/member hash,wheel path/member hash/wheel hash and repository path/hash where present all repeat that row byte-for-byte.",
          "native and process": "absolute path, build-id discriminant/value, ELF hash, installed association or native-build output, launcher, controller request and launch values resolve one exact ELF/install/launch-evidence row. provider_class is exactly initial-native,installed-extension,blas,onednn,openmp,libc,libm,libstdcxx,libpython,dynamic-loader, or supervisor. The no-wheel _fp_control extension uses only native-build-extension, never installed-extension.",
          "runtime readback": "readback_schema is exactly temporac.native-launch-carrier-readback.v5, temporac.linux-fd-readback.v5, or temporac.cpython-initialization-readback.v4; stage is allowed by that schema; readback_bytes is positive and readback_sha256 resolves complete canonical bytes including LF; launch and profile digests equal the same launch evidence and FD/config profile.",
          "trusted generated": "Resolve discovery_output_root_sha256 and trusted_row_sha256 to one exact next trusted-index row. Every direct field equals its eleven-key preimage; embedded generator_origin is exactly the row generator_origin, hashes to generator_origin_sha256, and has kind cpython-stdlib,project-installed,or installed-wheel. Only the adjacent dedicated compile/exec pair may reuse this byte-identical origin."
        },
        "field_types": "Every *_sha256 and nonnull *_sha256_or_null is exactly 64 lowercase hex. Every *_path/absolute_path/provider_path/normalized_path/runtime_member_normalized_path uses the preserved strict path algorithm for its path class and is nonempty; distribution,version,module_name,provider_kind,provider_class,entrypoint_owner,stage,profile,source_type,dynamic_mapping_role are strict nonempty ASCII tokens. readback_bytes and generated_source_bytes are positive JSON integers; ordinal,dynamic_mapping_load_group_ordinal,dynamic_mapping_primary_raw_sequence and raw_parent_sequence are nonnegative JSON integers. trusted source_type is exactly str-utf8 or bytes and profile is pre-dispatch-import-plan-v5. build-id, installed-association and dynamic-mapping null/present fields obey null_discriminants. No numeric string, omitted null, Unicode path alias, case fold or implicit token conversion is accepted.",
        "kind_closed_keys": {
          "cpython-core": [
            "kind",
            "module_name",
            "provider_kind",
            "provider_path",
            "provider_sha256",
            "runtime_member_normalized_path",
            "schema"
          ],
          "cpython-extension": [
            "absolute_path",
            "build_id_or_null",
            "build_id_tag",
            "dynamic_mapping_delta_sha256_or_null",
            "dynamic_mapping_load_group_ordinal",
            "dynamic_mapping_primary_raw_sequence",
            "dynamic_mapping_role",
            "elf_sha256",
            "kind",
            "provider_path",
            "runtime_member_normalized_path",
            "runtime_member_sha256",
            "schema"
          ],
          "cpython-stdlib": [
            "kind",
            "normalized_path",
            "provider_path",
            "runtime_member_sha256",
            "schema"
          ],
          "ctypes-dynamic-load": [
            "absolute_path",
            "build_id_or_null",
            "build_id_tag",
            "dynamic_mapping_delta_sha256",
            "dynamic_mapping_load_group_ordinal",
            "elf_sha256",
            "installed_member_association_sha256_or_null",
            "installed_member_association_tag",
            "kind",
            "provider_class",
            "raw_parent_sequence",
            "schema"
          ],
          "dispatch-boundary": [
            "entrypoint_owner",
            "kind",
            "schema",
            "source_allowlist_sha256"
          ],
          "installed-extension": [
            "absolute_path",
            "association_sha256",
            "build_id_or_null",
            "build_id_tag",
            "distribution",
            "dynamic_mapping_delta_sha256_or_null",
            "dynamic_mapping_load_group_ordinal",
            "dynamic_mapping_primary_raw_sequence",
            "dynamic_mapping_role",
            "elf_sha256",
            "installed_member_sha256",
            "installed_path",
            "kind",
            "schema",
            "version",
            "wheel_member_path",
            "wheel_member_sha256",
            "wheel_sha256"
          ],
          "installed-wheel": [
            "association_sha256",
            "distribution",
            "installed_member_sha256",
            "installed_path",
            "kind",
            "schema",
            "version",
            "wheel_member_path",
            "wheel_member_sha256",
            "wheel_sha256"
          ],
          "native-build-extension": [
            "absolute_path",
            "build_id_or_null",
            "build_id_tag",
            "build_output_sha256",
            "dynamic_mapping_delta_sha256_or_null",
            "dynamic_mapping_load_group_ordinal",
            "dynamic_mapping_primary_raw_sequence",
            "dynamic_mapping_role",
            "elf_sha256",
            "installed_member_sha256",
            "kind",
            "native_install_index_sha256",
            "schema"
          ],
          "native-image": [
            "absolute_path",
            "build_id_or_null",
            "build_id_tag",
            "elf_sha256",
            "installed_member_association_sha256_or_null",
            "installed_member_association_tag",
            "kind",
            "provider_class",
            "schema"
          ],
          "process-image": [
            "absolute_path",
            "build_id_or_null",
            "build_id_tag",
            "controller_request_sha256",
            "elf_sha256",
            "kind",
            "launch_instance_sha256",
            "launcher_sha256",
            "schema"
          ],
          "project-installed": [
            "association_sha256",
            "distribution",
            "installed_member_sha256",
            "installed_path",
            "kind",
            "repository_path",
            "repository_sha256",
            "schema",
            "version",
            "wheel_member_path",
            "wheel_member_sha256",
            "wheel_sha256"
          ],
          "runtime-readback": [
            "kind",
            "launch_instance_sha256",
            "profile_sha256",
            "readback_bytes",
            "readback_schema",
            "readback_sha256",
            "schema",
            "stage"
          ],
          "trusted-generated": [
            "compile_filename",
            "discovery_output_root_sha256",
            "generated_source_bytes",
            "generated_source_sha256",
            "generator_origin",
            "generator_origin_sha256",
            "kind",
            "ordinal",
            "profile",
            "schema",
            "source_type",
            "trusted_row_sha256"
          ]
        },
        "null_discriminants": {
          "build_id": "build_id_tag is exactly absent or present. absent requires build_id_or_null=null; present requires lowercase nonempty even-length hex equal the unique ELF build-id bytes.",
          "dynamic mapping": "For each extension origin dynamic_mapping_role is primary or mirror and dynamic_mapping_load_group_ordinal is always present. primary requires dynamic_mapping_delta_sha256_or_null present and dynamic_mapping_primary_raw_sequence equal the same normalized row's raw reference; mirror requires the digest null and the sequence equal its logical extension primary raw reference. Both group ordinals equal the exact expected group. ctypes-dynamic-load has no nullable group/delta fields. Every primary in one group repeats the same delta digest and occupies exactly one aligned position in its arrays.",
          "native installed association": "installed_member_association_tag is exactly no-installed-member or installed-member. no-installed-member requires digest null and provider_class not installed-extension; installed-member requires a lowercase digest resolving one exact association and provider_class installed-extension."
        },
        "schema_values": {
          "cpython-core": "temporac.executable-origin.cpython-core.v5",
          "cpython-extension": "temporac.executable-origin.cpython-extension.v5",
          "cpython-stdlib": "temporac.executable-origin.cpython-stdlib.v5",
          "ctypes-dynamic-load": "temporac.executable-origin.ctypes-dynamic-load.v5",
          "dispatch-boundary": "temporac.executable-origin.dispatch-boundary.v5",
          "installed-extension": "temporac.executable-origin.installed-extension.v5",
          "installed-wheel": "temporac.executable-origin.installed-wheel.v5",
          "native-build-extension": "temporac.executable-origin.native-build-extension.v5",
          "native-image": "temporac.executable-origin.native-image.v5",
          "process-image": "temporac.executable-origin.process-image.v5",
          "project-installed": "temporac.executable-origin.project-installed.v5",
          "runtime-readback": "temporac.executable-origin.runtime-readback.v5",
          "trusted-generated": "temporac.executable-origin.trusted-generated.v5"
        },
        "trusted_generated_projection": {
          "fixed_values": "origin.kind is trusted-generated and origin.schema is temporac.executable-origin.trusted-generated.v5. The resolved row kind is non-file-backed-python-source, its schema is temporac.trusted-generated-source-row.v5, and its profile is pre-dispatch-import-plan-v5.",
          "lookup": "Resolve discovery_output_root_sha256 to the exact accepted common twin-discovery output and trusted_row_sha256 to exactly one row of the committed trusted-generated-source index in ordinal order. Remove only trusted_row_sha256 from that stored twelve-key row, serialize the remaining exact eleven keys as canonical compact recursively sorted UTF-8 JSON plus one LF, and require SHA-256 equal trusted_row_sha256. No digest-only or reconstructed-from-trace substitute is accepted.",
          "one_to_one_fields": [
            "origin.compile_filename == row.compile_filename",
            "origin.generated_source_bytes == row.generated_source_bytes",
            "origin.generated_source_sha256 == row.generated_source_sha256",
            "canonical(origin.generator_origin)+LF == canonical(row.generator_origin)+LF",
            "origin.generator_origin_sha256 == row.generator_origin_sha256",
            "origin.ordinal == row.ordinal",
            "origin.profile == row.profile",
            "origin.source_type == row.source_type"
          ],
          "row_only_validation": "The resolved row generated_source_hex decodes to exactly generated_source_bytes and hashes to generated_source_sha256. Its generator_origin is one exact cpython-stdlib, project-installed, or installed-wheel v5 origin, hashes to generator_origin_sha256 under the same origin preimage rule, and identifies the active source/installed ledger row. The origin deliberately omits generated_source_hex and row kind; those values are recovered only through trusted_row_sha256 and must pass this exact projection before the adjacent compile/exec pair is authorized."
        },
        "variant_rule": "origin.kind must equal exactly one kind_closed_keys key, origin.schema must equal its schema_values value, and the object keys must equal that kind's list. Every direct path uses the one preserved strict normalization algorithm. PY_CODE_EVAL has an empty allowed-kind list and exact multiplicity zero. No v4 object is accepted by substitution; v4 scientific artifacts remain historical inputs, while only this v5 union represents Round7 native trace rows."
      },
      "phase_state_machine": "Discovery and full runtime share the same raw catalog and projections but not authority. Discovery may construct a candidate only from the current raw compile bytes, exact file-backed generator frame and immediately adjacent exec; the candidate is appended to that child's output and cannot authorize that child. Twin output equality plus joined evidence creates the trusted index. A fresh full child starts with that committed index, and every generated compile must match the next row before execution; all rows are exhausted before DISPATCH_BEGIN. No discovery output is fed back into its own launch, BASE, capsule or precommit.",
      "raw_row_closed_keys": [
        "args",
        "args_sha256",
        "event",
        "process_ordinal",
        "schema",
        "sequence"
      ],
      "raw_row_schema": "temporac.raw-audit-event.v5",
      "raw_serialization": "schema is exactly temporac.raw-audit-event.v5. args is the event-specific closed typed projection named by the catalog. It contains no repr, pointer, traceback, arbitrary object JSON, inferred compile mode or provider guess. args_sha256 hashes the standalone canonical args object including LF; process_ordinal is nonnegative, sequence is zero-based/gap-free per process, and event exactly equals the matched catalog event_name.",
      "safe_barriers": [
        "immediately after successful Py_InitializeFromConfig and before post-initialize FD/config readback",
        "immediately before the supervisor calls each exact requested top-level PyImport_ImportModule",
        "successful return from that same call to the supervisor-controlled import loop",
        "immediately before DISPATCH_BEGIN after every expected load group is exhausted",
        "immediately after Py_FinalizeEx for terminal mapping capture"
      ],
      "subprocess_coverage": "Only the native controller may fork, and every permitted child immediately execve's the same absolute supervisor with the exact child argv. The controller assigns process_ordinal from the request mode and reserves normalized sequence zero for PROCESS_EXEC, but constructs that row only after the same child's carrier-validated sequence-one row, PROCESS_IMAGE row, terminal EOF and zero-exit waitpid jointly prove the exec. The child has no launch identity before carrier validation and emits no implicit handshake. Recursive fork/exec, Python subprocess, shell, PATH lookup, fork-without-exec, alternate image, ordinal collision, missing carrier/image/join evidence, unjoined child or executable descendant is forbidden.",
      "uniqueness": "Source, installed-wheel, CPython, native-install, ELF, runtime-readback, generated and dispatch authority keys are unique. An origin object is the sole canonical representation of one identity; only source-to-compile-to-exec and trusted-compile-to-exec chains may deliberately reuse a byte-identical origin. Event rows remain unique by process_ordinal/sequence and raw references; same digest with different bytes is a collision failure."
    },
    "build_and_deploy": {
      "build_environment": {
        "capabilities": [],
        "cwd_rule": "Each build-exec row binds its exact cwd. source_extract and every project-native outer command use /build/temporac; CPython configure,make,install and their descendants use /build/temporac/cpython-build unless the independently accepted exec plan names an exact child cwd beneath that build tree. No process consults inherited cwd by omission.",
        "envp": [
          "AR=/opt/temporac/toolchain/bin/x86_64-linux-gnu-ar",
          "CC=/opt/temporac/toolchain/bin/x86_64-linux-gnu-gcc",
          "CFLAGS=-O2 -g0 -march=x86-64-v2 -mtune=generic -fno-fast-math -ffp-contract=off -frounding-math -fexcess-precision=standard -fno-strict-aliasing -fno-omit-frame-pointer -fno-plt -fno-common -fno-ident -ffile-prefix-map=/build/temporac=. -fdebug-prefix-map=/build/temporac=. -fmacro-prefix-map=/build/temporac=. -fstack-protector-strong",
          "CPPFLAGS=-D_FORTIFY_SOURCE=3",
          "HOME=/build/temporac/home",
          "LANG=C",
          "LC_ALL=C",
          "LDFLAGS=-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=sha1 -Wl,--disable-new-dtags -Wl,-rpath,/opt/temporac/replay-v4/lib",
          "MAKEFLAGS=-j1",
          "PATH=/opt/temporac/toolchain/bin:/usr/bin:/bin",
          "PYTHONHASHSEED=0",
          "PYTHONNOUSERSITE=1",
          "PYTHONDONTWRITEBYTECODE=1",
          "SOURCE_DATE_EPOCH=315532800",
          "TMPDIR=/build/temporac/tmp",
          "TZ=UTC",
          "XDG_CACHE_HOME=/build/temporac/cache",
          "ZERO_AR_DATE=1"
        ],
        "gid": 65532,
        "network": "disabled",
        "no_new_privs": 1,
        "supplementary_groups": [],
        "uid": 65532,
        "umask_octal": "077"
      },
      "commands": {
        "fp_compile": [
          "/opt/temporac/toolchain/bin/x86_64-linux-gnu-gcc",
          "-std=c17",
          "-O2",
          "-g0",
          "-march=x86-64-v2",
          "-mtune=generic",
          "-fno-fast-math",
          "-ffp-contract=off",
          "-frounding-math",
          "-fexcess-precision=standard",
          "-fno-strict-aliasing",
          "-fno-omit-frame-pointer",
          "-fno-plt",
          "-fno-common",
          "-fno-ident",
          "-ffile-prefix-map=/build/temporac=.",
          "-fdebug-prefix-map=/build/temporac=.",
          "-fmacro-prefix-map=/build/temporac=.",
          "-fstack-protector-strong",
          "-D_FORTIFY_SOURCE=3",
          "-Wall",
          "-Wextra",
          "-Werror",
          "-fPIC",
          "-fvisibility=hidden",
          "-MD",
          "-MF",
          "/build/temporac/out/_fp_control.d",
          "-MT",
          "/build/temporac/out/_fp_control.o",
          "-DPAMS_TEMPORAC_FP_EXTENSION=1",
          "-I/build/temporac/cpython-stage/opt/temporac/replay-v4/include/python3.12",
          "-c",
          "/build/temporac/src/pams/temporac/_fp_control.c",
          "-o",
          "/build/temporac/out/_fp_control.o"
        ],
        "fp_link": [
          "/opt/temporac/toolchain/bin/x86_64-linux-gnu-gcc",
          "-shared",
          "-Wl,-z,relro",
          "-Wl,-z,now",
          "-Wl,-z,noexecstack",
          "-Wl,--build-id=sha1",
          "-Wl,--no-undefined",
          "-Wl,--disable-new-dtags",
          "-Wl,-rpath,/opt/temporac/replay-v4/lib",
          "-L/build/temporac/cpython-stage/opt/temporac/replay-v4/lib",
          "-o",
          "/build/temporac/out/_fp_control.cpython-312-x86_64-linux-gnu.so",
          "/build/temporac/out/_fp_control.o",
          "-lpython3.12",
          "-ldl",
          "-lm",
          "-lpthread"
        ],
        "fp_native_compile": [
          "/opt/temporac/toolchain/bin/x86_64-linux-gnu-gcc",
          "-std=c17",
          "-O2",
          "-g0",
          "-march=x86-64-v2",
          "-mtune=generic",
          "-fno-fast-math",
          "-ffp-contract=off",
          "-frounding-math",
          "-fexcess-precision=standard",
          "-fno-strict-aliasing",
          "-fno-omit-frame-pointer",
          "-fno-plt",
          "-fno-common",
          "-fno-ident",
          "-ffile-prefix-map=/build/temporac=.",
          "-fdebug-prefix-map=/build/temporac=.",
          "-fmacro-prefix-map=/build/temporac=.",
          "-fstack-protector-strong",
          "-D_FORTIFY_SOURCE=3",
          "-Wall",
          "-Wextra",
          "-Werror",
          "-fPIE",
          "-fvisibility=hidden",
          "-MD",
          "-MF",
          "/build/temporac/out/_fp_control_native.d",
          "-MT",
          "/build/temporac/out/_fp_control_native.o",
          "-DPAMS_TEMPORAC_FP_NATIVE=1",
          "-c",
          "/build/temporac/src/pams/temporac/_fp_control.c",
          "-o",
          "/build/temporac/out/_fp_control_native.o"
        ],
        "project_wheel": [
          "/opt/temporac/build/bin/python3.12",
          "-P",
          "-s",
          "-m",
          "hatchling",
          "build",
          "-t",
          "wheel",
          "-d",
          "/build/temporac/out"
        ],
        "source_extract": [
          "/opt/temporac/toolchain/bin/bsdtar",
          "--extract",
          "--file",
          "/build/temporac/input/Python-3.12.13.tar.xz",
          "--directory",
          "/build/temporac/source",
          "--no-same-owner",
          "--no-same-permissions"
        ],
        "supervisor_compile": [
          "/opt/temporac/toolchain/bin/x86_64-linux-gnu-gcc",
          "-std=c17",
          "-O2",
          "-g0",
          "-march=x86-64-v2",
          "-mtune=generic",
          "-fno-fast-math",
          "-ffp-contract=off",
          "-frounding-math",
          "-fexcess-precision=standard",
          "-fno-strict-aliasing",
          "-fno-omit-frame-pointer",
          "-fno-plt",
          "-fno-common",
          "-fno-ident",
          "-ffile-prefix-map=/build/temporac=.",
          "-fdebug-prefix-map=/build/temporac=.",
          "-fmacro-prefix-map=/build/temporac=.",
          "-fstack-protector-strong",
          "-D_FORTIFY_SOURCE=3",
          "-Wall",
          "-Wextra",
          "-Werror",
          "-fPIE",
          "-MD",
          "-MF",
          "/build/temporac/out/_origin_supervisor.d",
          "-MT",
          "/build/temporac/out/_origin_supervisor.o",
          "-DPAMS_TEMPORAC_SUPERVISOR=1",
          "-I/build/temporac/cpython-stage/opt/temporac/replay-v4/include/python3.12",
          "-c",
          "/build/temporac/src/pams/temporac/_origin_supervisor.c",
          "-o",
          "/build/temporac/out/_origin_supervisor.o"
        ],
        "supervisor_link": [
          "/opt/temporac/toolchain/bin/x86_64-linux-gnu-gcc",
          "-pie",
          "-Wl,-z,relro",
          "-Wl,-z,now",
          "-Wl,-z,noexecstack",
          "-Wl,--build-id=sha1",
          "-Wl,--no-undefined",
          "-Wl,--disable-new-dtags",
          "-Wl,-rpath,/opt/temporac/replay-v4/lib",
          "-L/build/temporac/cpython-stage/opt/temporac/replay-v4/lib",
          "-o",
          "/build/temporac/out/temporac-origin-supervisor",
          "/build/temporac/out/_origin_supervisor.o",
          "/build/temporac/out/_fp_control_native.o",
          "-lpython3.12",
          "-ldl",
          "-lm",
          "-lpthread"
        ]
      },
      "cpython": {
        "build_rule": "Build CPython twice from the exact source archive in separate empty roots with the selected compiler, single-threaded make, network disabled, no PGO/LTO/ensurepip, then require every installed file, ELF member, wheel and ledger byte-identical before either build may populate the runtime index.",
        "build_runs": [
          {
            "build_cwd": "/build/temporac/cpython-build",
            "configure_argv": [
              "/bin/bash",
              "--noprofile",
              "--norc",
              "/build/temporac/source/Python-3.12.13/configure",
              "--prefix=/opt/temporac/replay-v4",
              "--exec-prefix=/opt/temporac/replay-v4",
              "--enable-shared",
              "--without-ensurepip",
              "--with-pymalloc",
              "--with-platlibdir=lib",
              "--with-lto=no",
              "--disable-test-modules",
              "--with-hash-algorithm=siphash13"
            ],
            "install_argv": [
              "/usr/bin/make",
              "-j1",
              "DESTDIR=/build/temporac/cpython-stage",
              "install"
            ],
            "isolated_instance": "clean-image-instance-0",
            "make_argv": [
              "/usr/bin/make",
              "-j1",
              "all"
            ],
            "ordinal": 0,
            "stage_export_role": "cpython-stage-export",
            "stage_root": "/build/temporac/cpython-stage"
          },
          {
            "build_cwd": "/build/temporac/cpython-build",
            "configure_argv": [
              "/bin/bash",
              "--noprofile",
              "--norc",
              "/build/temporac/source/Python-3.12.13/configure",
              "--prefix=/opt/temporac/replay-v4",
              "--exec-prefix=/opt/temporac/replay-v4",
              "--enable-shared",
              "--without-ensurepip",
              "--with-pymalloc",
              "--with-platlibdir=lib",
              "--with-lto=no",
              "--disable-test-modules",
              "--with-hash-algorithm=siphash13"
            ],
            "install_argv": [
              "/usr/bin/make",
              "-j1",
              "DESTDIR=/build/temporac/cpython-stage",
              "install"
            ],
            "isolated_instance": "clean-image-instance-1",
            "make_argv": [
              "/usr/bin/make",
              "-j1",
              "all"
            ],
            "ordinal": 1,
            "stage_export_role": "cpython-stage-export",
            "stage_root": "/build/temporac/cpython-stage"
          }
        ],
        "build_runs_rule": "Run the two rows in two separately created, nonoverlapping writable instances of the same immutable reviewed build image. In each mount namespace create /build/temporac root0:0 mode0755, nonwritable to uid65532; precreate only the root-owned input/source mountpoints and require cpython-build,cpython-stage,tmp,cache,home initially absent. Materialize the exact 20801708-byte source archive in a verified root-owned lower object, bind it at /build/temporac/input/Python-3.12.13.tar.xz as regular 0444 under input directory 0555 with bind,ro,nosuid,nodev,noexec and read back mountinfo/inode/bytes/hash. Create a separate temporary extraction root owner65532:65532 mode0700, execute source_extract exactly once as uid65532, require one top-level Python-3.12.13 tree, validate every extracted path/kind/byte against the archive, then have the root controller rematerialize that complete tree into a new root0:0 lower tree with directories0555, regular files0444 and exact relative symlinks; bind that lower tree read-only,nosuid,nodev,noexec at /build/temporac/source and destroy/prove-inaccessible the writable extraction root before configure. Create cpython-build,cpython-stage,tmp,cache owner65532:65532 mode0700 and home owner65532:65532 mode0500 under umask077. Exactly cpython-build,cpython-stage,tmp,cache are writable to uid65532; home is nonwritable mode0500, and every other named input/source/tool root is root-owned and read-only. Toolchain/build-image mounts are the exact precommitted read-only executable ledgers. Execute every command as uid=gid=65532 with supplementary groups/capabilities empty and no_new_privs=1; all temporary/cache writes stay under exact TMPDIR/XDG paths. The instances share no writable layer, pid namespace, cache, temp directory or output path and have network disabled; isolated_instance is external evidence only, while both build-output indexes use the same role token cpython-stage-export and are distinguished solely by run_ordinal. Execute configure then make then install in the identical cwd shown. argv tokens are direct with no shell expansion other than fixed bash executing the exact read-only configure script. Strip, ensurepip, network and post-install mutation are forbidden. Deterministic checked-hash stdlib pyc files are allowed only when complete bytes appear identically in both stage-export ledgers and the exact exclusion index; none is deployed and runtime write_bytecode remains zero. After install, reopen and rehash the archive, complete source mount, mount flags, tool/build-image ledger and every immutable input; require exact pre-run equality before accepting any output. Capture complete command streams, exec indexes, cpython-build/stage output indexes and tmp/cache/home indexes for both runs, normalize only declared run_ordinal fields, and require byte identity under native_rebuild_evidence before export0 can deploy.",
        "install_exclusion_index": {
          "digest_rule": "cpython_install_exclusion_index_sha256 is SHA-256 of the complete canonical index bytes including LF and is absent from the index. The exact bytes are retained under native_rebuild_evidence.object_store_rule.",
          "row_closed_keys": [
            "bytes",
            "kind",
            "path",
            "reason",
            "sha256"
          ],
          "rule": "The future artifact has exactly top_level_closed_keys. native_rebuild_evidence_index_sha256 resolves the accepted four-run family and both CPython stage-export projections. rows are one unique strict ASCII path-ordered projection from run0 cpython-stage-export output rows: exactly every staged regular .pyc member plus every __pycache__ directory that becomes empty after all its excluded pyc children are removed; reason is no-runtime-marshaled-code. A source output row is eligible only when its path begins exact prefix opt/temporac/replay-v4/; exclusion path is the nonempty suffix after removing that one prefix, never the full stage path, and the prefix root row itself is never excludable. Run1 must have byte-identical same projection after ordinal normalization and contributes no duplicate. kind is regular or directory; regular bytes/hash equal run0 output, directory has bytes=0 and sha256=SHA-256(empty). Native-install/deployment uses the same suffix namespace: each nonexcluded output prefix+suffix maps exactly to installed /opt/temporac/replay-v4/suffix, while the prefix root maps only to installed root. No source,extension,data,header,library or other member may appear. Deployment copies exact run0 complement without mutating retained stage trees.",
          "schema": "temporac.cpython-install-exclusion-index.v5",
          "top_level_closed_keys": [
            "contract_sha256",
            "native_rebuild_evidence_index_sha256",
            "rows",
            "schema"
          ]
        },
        "source_archive_bytes": 20801708,
        "source_archive_sha256": "c08bc65a81971c1dd5783182826503369466c7e67374d1646519adf05207b684",
        "source_archive_url": "https://www.python.org/ftp/python/3.12.13/Python-3.12.13.tar.xz",
        "source_extract_mount_transition": "Immediately before source_extract, bind-mount the fresh owner65532:65532 mode0700 writable extraction root onto the already precreated /build/temporac/source mountpoint with exact bind,nosuid,nodev,noexec flags and read back mount/inode/writable ownership; the fixed --directory path therefore names that writable root. After bsdtar exits0,close/rehash and validate the extracted tree, unmount that binding, require the root-owned mountpoint empty, rematerialize into the root0:0 read-only lower tree, then bind that lower tree at the same path with bind,ro,nosuid,nodev,noexec and read back. Destroy and prove inaccessible the original writable extraction root before configure. No command observes an alternate source path and no two source mounts coexist.",
        "version": "3.12.13",
        "workspace_directory_creation_rule": "This exact rule controls the directory sequence in build_runs_rule: initially precreate as root-owned only /build/temporac/input and /build/temporac/source mountpoints. Require cpython-build,cpython-stage,tmp,cache,home absent. Then create exactly once cpython-build,cpython-stage,tmp,cache owner65532:65532 mode0700 and home owner65532:65532 mode0500 under umask077. Exactly the first four created directories are writable to uid65532; home is nonwritable. Wrong preexistence,owner,mode,writability or second creation fails."
      },
      "deployment": {
        "algorithm": [
          "Create exact empty /opt/temporac/replay-v4.new with owner 0:0, mode 0755 and umask 077; existing final or staging path fails, and no symlink is followed.",
          "Copy the exact run-0 CPython staged-member complement of the validated cpython-install-exclusion index from its /opt/temporac/replay-v4 prefix into the staging root with EINTR-safe reads/writes, preserving only the ledgered regular-file/directory/symlink kind, relative path, mode and symlink target; device, socket, fifo, hard-link alias and unlisted metadata fail. No staged pyc or __pycache__ directory is deployed.",
          "Iterate the exact accepted project-wheel plus every third-party wheel-index row in the index's frozen distribution/version/filename/hash order. For each exact wheel, verify archive bytes/hash, parse its ZIP central directory with duplicate-name rejection and the frozen wheel path/scheme mapping, and in exact member order decompress each file once to its predicted purelib, platlib, scripts, headers or data destination under the staging prefix. Every destination path/kind/mode/bytes/hash must equal the corresponding native-install and installed-member association row before write; cross-wheel collision is allowed only for an explicit byte-identical shared-directory row and never for a file. No installer code, dependency resolution, network, entry-point wrapper, RECORD rewrite, INSTALLER/direct_url metadata, timestamp, uid/gid, xattr or pyc is created.",
          "Install the exact supervisor build output as bin/temporac-origin-supervisor mode 0755 and _fp_control output at lib/python3.12/site-packages/pams/temporac/_fp_control.cpython-312-x86_64-linux-gnu.so mode 0644; both are native-build-output/no-wheel-member and byte-equal their reviewed build-output ledger rows.",
          "Generate no byte during deployment. Build a complete staging member ledger; require it byte-for-byte equal the independently predicted native-install index, fsync every regular file and directory bottom-up, renameat2 staging to final with RENAME_NOREPLACE, fsync /opt/temporac, reopen every final member without following links, and rehash the complete ledger. Any preexisting final path or equality failure aborts without publishing a runtime."
        ],
        "installed_root": "/opt/temporac/replay-v4",
        "rule": "The native-install index is the sole deployment plan and complete resulting member roster. It binds both equal CPython build ledgers, the project wheel and every exact accepted third-party wheel/member association required by the complete import closure, two native outputs, modes, owners, link targets, ELF acceptance and final paths. After CPython and all wheel rows, only supervisor and _fp_control may overlay, and both require a previously absent exact destination. The final staging member ledger must equal the predicted native-install index with no missing or extra distribution member. No package installer, editable source, dependency resolver, network, PATH tool, shell glob or current-tree import participates.",
        "staging_root": "/opt/temporac/replay-v4.new"
      },
      "elf_acceptance": {
        "closed_row_keys": [
          "build_id_hex_or_null",
          "dt_needed",
          "elf_sha256",
          "elf_type",
          "gnu_relro",
          "has_executable_stack",
          "installed_path",
          "machine",
          "pie_or_pic",
          "rpath_or_null",
          "runpath_or_null",
          "symbol_strip_state",
          "text_relocations",
          "z_now"
        ],
        "required": "Supervisor is ET_DYN PIE; _fp_control and libpython are ET_DYN PIC; machine is EM_X86_64; GNU_RELRO=true, z_now=true, has_executable_stack=false, text_relocations=false, symbol_strip_state=unstripped, rpath_or_null=/opt/temporac/replay-v4/lib, runpath_or_null=null, and build_id_hex_or_null is exactly the lowercase 20-byte SHA-1 build-id emitted by the selected linker. dt_needed is the exact ordered DT_NEEDED string array from the ELF and must resolve bijectively through the complete ELF ledger. No unknown dynamic tag, post-link patch, different SONAME/provider or loader search fallback is accepted.",
        "schema": "temporac.native-elf-acceptance-index.v5"
      },
      "forbidden": [
        "shell expansion in the exact five direct native compile/link commands supervisor_compile,fp_native_compile,supervisor_link,fp_compile,fp_link; the only shell use is the exact argv execution of CPython's bound configure script",
        "python-config or pkg-config discovery",
        "PATH compiler/linker selection or any PATH-resolved build image absent from the preaccepted build-exec/tool ledgers",
        "LTO, PGO, fast-math, contraction, native march, stripping or post-link mutation",
        "editable project install, project console-script synthesis, project RECORD rewrite, runtime pyc generation or current-tree import"
      ],
      "fp_native_abi": {
        "call_order": "After FD8/profile validation and locked-pre-cpython FD readback but before PySys_AddAuditHook or any CPython API, the embedded supervisor calls set_round(FE_TONEAREST), set_mxcsr(0x00001f80), get_round and get_mxcsr through the statically linked native ABI and requires exact success/readback. The later Python extension wrappers invoke the same source-level operations for the preserved Torch/probe checks. Every reset/readback point in floating_point_control remains exact.",
        "compile_modes": "The one allowlisted _fp_control.c source must compile in exactly two mutually exclusive modes. PAMS_TEMPORAC_FP_NATIVE=1 emits the five hidden C ABI definitions and no PyInit/Python reference; PAMS_TEMPORAC_FP_EXTENSION=1 emits PyInit__fp_control plus the exact five Python-callable wrappers and no second arithmetic implementation. Defining both/neither, conditional host probing, generated source or semantic drift fails.",
        "floating_point_profile_overlay": {
          "artifact_rule": "Construct the BASE floating-point-control-profile role by taking the exact accepted Round6 floating_point_control object, replacing only json_pointer with the complete literal value below, then serializing that complete object as strict canonical JSON with one LF. The row sha256 hashes those exact bytes. Every other helper field, export, call, readback, reset, Torch flush-denormal and failure rule remains byte-for-byte normative; hashing the unmodified Round6 object, omitting the replacement, adding a second replacement or treating the native output as a wheel member fails.",
          "json_pointer": "/cpu_replay_environment/floating_point_control/helper/origin",
          "value": "the exact /opt/temporac/replay-v4/lib/python3.12/site-packages/pams/temporac/_fp_control.cpython-312-x86_64-linux-gnu.so native-build-output/no-wheel-member whose complete bytes/hash resolve one native-build output, native-install-index row and ELF row and whose sole permitted executable-origin kind is native-build-extension with one PY_EXTENSION_LOAD event"
        },
        "native_signatures": [
          "int temporac_fp_fe_tonearest(void)",
          "int temporac_fp_get_round(void)",
          "int temporac_fp_set_round(int mode)",
          "uint32_t temporac_fp_get_mxcsr(void)",
          "int temporac_fp_set_mxcsr(uint32_t value)"
        ],
        "return_rules": "fe_tonearest returns the exact platform FE_TONEAREST int. get_round returns fegetround. set_round returns the unmodified fesetround result. get_mxcsr returns exact _mm_getcsr bits. set_mxcsr performs _mm_setcsr(value), immediately requires _mm_getcsr()==value and returns 0 on equality or -1 otherwise. The supervisor accepts only the exact required success/readbacks and maps failure to FP_LOCK.",
        "symbol_rule": "The native object is linked only into the supervisor PIE and its ABI symbols have ELF hidden visibility; it creates no separately installed member. The extension object is linked only into the exact _fp_control shared object; its only default-visible module initializer is PyInit__fp_control and its Python module exposes exactly the preserved five closed exports. Both object/depfile bytes and the supervisor/extension outputs are covered by the twin rebuild ledger."
      },
      "future_populated_fields": "Compiler, assembler, linker, every build-exec image/argv row, libc, headers, crt objects, libpython, build image, Hatchling dependency closure, all command inputs, all four non-authoritative preflight evidence families, all four authority-run rebuild outputs, installed outputs, build IDs, DT_NEEDED and every byte count/hash are mandatory exact non-null P00/P06 ledger values. The preflight instances execute in the frozen order CPython0,CPython1,project-native0,project-native1; the project pair receives only the byte-equal validated CPython0 preflight stage. Their complete retained pipeline and single two-kind plan require independent acceptance before all four authority-bearing twin runs. This proposal assigns no fabricated value and no execution may begin while any value or acceptance is absent.",
      "hatchling": {
        "selected_version": "1.27.0",
        "wheel_sha256": "d3a2f3567c4f926ea39849cdf924c7e99e6686c9c8e288ae1037c8fa2a5d937b"
      },
      "install_associations": {
        "fp_extension": {
          "association_kind": "native-build-output",
          "installed_path": "/opt/temporac/replay-v4/lib/python3.12/site-packages/pams/temporac/_fp_control.cpython-312-x86_64-linux-gnu.so",
          "member_tag": "no-wheel-member",
          "repository_path_or_null": null,
          "repository_sha256_or_null": null,
          "repository_tag": "no-repository-member",
          "schema": "temporac.native-output-install-association.v5",
          "wheel_member_path_or_null": null,
          "wheel_member_sha256_or_null": null,
          "wheel_sha256_or_null": null
        },
        "native_output_closed_keys": [
          "association_kind",
          "installed_path",
          "member_tag",
          "repository_path_or_null",
          "repository_sha256_or_null",
          "repository_tag",
          "schema",
          "wheel_member_path_or_null",
          "wheel_member_sha256_or_null",
          "wheel_sha256_or_null"
        ],
        "native_output_rule": "Each native output object has exactly native_output_closed_keys. association_kind=native-build-output, member_tag=no-wheel-member, repository_tag=no-repository-member, every repository/wheel nullable value is null, schema is temporac.native-output-install-association.v5, and installed_path equals the unique native-install-index row. Its actual bytes/hash/build-output/ELF identity are mandatory in that future row and are not fabricated in this static path template. This v5 object is never parsed as the wheel-only temporac.installed-member-association.v4.",
        "project_python": "Every installed project Python byte must equal exactly one repository member and one noneditable project-wheel association. Every such byte that the import plan or trace imports/executes, including future pams/temporac/protocol_entry.py, must additionally equal exactly one precommitted source-allowlist row; an installed non-allowlisted project member must have zero import/compile/exec events.",
        "supervisor": {
          "association_kind": "native-build-output",
          "installed_path": "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
          "member_tag": "no-wheel-member",
          "repository_path_or_null": null,
          "repository_sha256_or_null": null,
          "repository_tag": "no-repository-member",
          "schema": "temporac.native-output-install-association.v5",
          "wheel_member_path_or_null": null,
          "wheel_member_sha256_or_null": null,
          "wheel_sha256_or_null": null
        }
      },
      "native_install_index": {
        "hash_rule": "native_install_index_sha256 is SHA-256 of the complete canonical index bytes including LF. native_rebuild_evidence_index_sha256 resolves the accepted twin evidence before this index is constructed. Every installed path is unique and rows are strict ASCII installed_path order. The index has no own digest and is the sole predicted/deployed member roster.",
        "row_closed_keys": [
          "association_kind",
          "build_output_sha256_or_null",
          "bytes",
          "distribution_or_null",
          "gid",
          "installed_path",
          "kind",
          "link_target_or_null",
          "mode_octal",
          "repository_path_or_null",
          "repository_sha256_or_null",
          "sha256_or_null",
          "uid",
          "wheel_member_path_or_null",
          "wheel_member_sha256_or_null",
          "wheel_sha256_or_null"
        ],
        "row_rule": "kind is regular,directory,or symlink. uid=0 and gid=0 for every row and are verified by lstat before staging acceptance and after final reopen. A regular has nonnegative bytes, lowercase sha256_or_null present over exact file bytes, and null link target; a directory has bytes=0 and all link/hash source fields null; a symlink has a strict relative nonempty link_target_or_null, bytes equal its UTF-8 count, sha256_or_null hash of those target bytes, and cannot escape the root. association_kind is cpython-stage,wheel-member,native-build-output,or directory. cpython-stage requires every build/wheel/repository/distribution field null and maps from exactly one cpython run0 stage-export row through the sole prefix transform in build_output_index.row_rule: source path opt/temporac/replay-v4[/suffix] becomes installed_path /opt/temporac/replay-v4[/suffix], while kind/mode/bytes/hash/target are exact and every stage container row above that prefix is excluded. The complete nonexcluded prefix projection is bijective, so no literal-path mismatch,digest-only or prose-only export deploys. wheel-member requires distribution and all three wheel fields present and build output null. Its repository pair is present iff the preserved installed-member association has repository_tag=repository-member and equals that source row; it is null for third-party/generated metadata. native-build-output requires regular, present build_output digest resolving one project-native run0 typed row; install bytes/hash equal output bytes/object hash while field is typed row digest. All distribution/wheel/repository fields are null. directory requires kind=directory and all source fields null unless exact cpython-stage projection. mode_octal is four ASCII octal digits and equals deployed lstat mode.",
        "schema": "temporac.native-install-index.v5",
        "top_level_closed_keys": [
          "contract_sha256",
          "cpython_install_exclusion_index_sha256",
          "native_rebuild_evidence_index_sha256",
          "rows",
          "schema",
          "wheel_index_sha256"
        ]
      },
      "native_rebuild_evidence": {
        "build_input_terminal_index": {
          "digest_rule": "build_input_pre_index_sha256 or build_input_terminal_index_sha256 is SHA-256 of the corresponding complete canonical snapshot bytes including LF and is absent from that index.",
          "row_closed_keys": [
            "bytes",
            "gid",
            "kind",
            "mode_octal",
            "mount_flags",
            "object_sha256_or_null",
            "path",
            "role",
            "run_ordinal",
            "symlink_target_or_null",
            "uid"
          ],
          "row_rule": "This schema is used twice per run. For cpython, phase=pre-run is captured after source_extract exits0, the extracted tree validates and is rematerialized at the read-only source mount, and before configure; source_extract itself remains the first retained outer command and is constrained by the archive, command-stream and build-exec evidence rather than by a nonexistent pre-extraction source-tree row. For project-native, phase=pre-run is captured after immutable mounts validate and before project_wheel. In both kinds phase=terminal is captured after the final command and before output acceptance. path is exact absolute visible namespace path without alias; role classifies. Rows strict role,path unique. build_kind cpython/project-native, run ordinal0/1 repeated. kind regular,directory,symlink; uid/gid/mode and mount_flags are exact lstat/nearest-mount readbacks. mount_flags is exactly sorted [bind,nodev,noexec,nosuid,ro] for source-archive,source-tree,project-source,project-build-input,cpython-stage,build-script and exactly [bind,nodev,nosuid,ro] for toolchain/build-image because executable tools require exec; no other flag/token or inherited writable mount is accepted. Regular has count/null target/hash; directory count0/nulls; symlink relative nonescaping target/count/hash. CPython covers archive,complete source,toolchain/build-image/scripts/config. Project-native covers 30 allowlist rows (29 src plus root pyproject), one README, complete run0 CPython stage,toolchain/build image,Hatchling. No output/tmp/cache/home. Within run terminal equals pre after only phase terminal→pre-run; across twins corresponding phase indexes equal after only top/row run ordinal→0. Mutation or accessible undeclared input fails.",
          "schema": "temporac.native-build-input-snapshot-index.v5",
          "top_level_closed_keys": [
            "build_kind",
            "contract_sha256",
            "phase",
            "rows",
            "run_ordinal",
            "schema"
          ]
        },
        "build_output_index": {
          "digest_rule": "build_output_index_sha256 is SHA-256 of complete canonical index bytes including LF and is absent from the index. build_output_sha256 is SHA-256 of the standalone canonical nine-key projection build_kind,bytes,kind,link_target_or_null,mode_octal,object_sha256,path,role,run_ordinal plus LF; it is then appended as the tenth stored key.",
          "row_closed_keys": [
            "build_kind",
            "build_output_sha256",
            "bytes",
            "kind",
            "link_target_or_null",
            "mode_octal",
            "object_sha256",
            "path",
            "role",
            "run_ordinal"
          ],
          "row_rule": "The index top-level build_kind/run_ordinal exactly equal the enclosing evidence row, and every stored row repeats them; mixed-run or mixed-kind rows fail. role is exactly cpython-build-tree or cpython-stage-export when build_kind=cpython and exactly project-out when build_kind=project-native. Its canonical root is respectively /build/temporac/cpython-build, /build/temporac/cpython-stage, or /build/temporac/out. path is '.' only for that root row; every child path is strict normalize_rel_posix_ascii relative to that root and cannot begin slash or escape. Rows are strict role,path byte order and role/path are unique. kind is depfile,directory,elf,object,regular,symlink,or zip. bytes is nonnegative; a directory has bytes=0,null target and object_sha256=SHA-256(empty); a symlink has a strict relative nonescaping target, bytes equal its UTF-8 count and object_sha256 over those target bytes; every other kind has null target and object_sha256 over complete bytes retained at object_store_root/objects/<object_sha256>.bin. mode_octal is exact four-digit lstat mode. CPython rows exhaust both canonical roots and every configure/make/install output/intermediate, including stage path '.', the container directories opt and opt/temporac, the installed prefix opt/temporac/replay-v4 and all descendants. Project-native rows exhaust project-out: project wheel, supervisor/extension ELFs, native objects, all three -MD depfiles and every other declared output/intermediate. The stored build_output_sha256 validates its exact nine-key preimage. A native-install native-build-output resolves exactly one project-out run0 regular/zip/elf row; installed bytes/hash equal that row bytes/object_sha256. The only deployable cpython-stage-export rows are path exactly opt/temporac/replay-v4 or beginning opt/temporac/replay-v4/; strip that one prefix and map its root to /opt/temporac/replay-v4 and a suffix p to /opt/temporac/replay-v4/p. Stage container rows '.',opt,opt/temporac remain retained output evidence but are not native-install members. No other prefix stripping, alias, output omission or cross-build role exists.",
          "schema": "temporac.native-build-output-index.v5",
          "top_level_closed_keys": [
            "build_kind",
            "contract_sha256",
            "rows",
            "run_ordinal",
            "schema"
          ]
        },
        "command_stream_index": {
          "digest_rule": "command_stream_index_sha256 is SHA-256 of the complete canonical index bytes including LF and is absent from the index. Each stdout/stderr digest hashes the exact retained stream bytes, including the unique empty-byte digest when its count is zero.",
          "row_closed_keys": [
            "command_token",
            "exit_code",
            "ordinal",
            "run_ordinal",
            "stderr_bytes",
            "stderr_sha256",
            "stdout_bytes",
            "stdout_sha256"
          ],
          "row_rule": "The index top-level build_kind/run_ordinal exactly equal the enclosing evidence row and every row repeats run_ordinal. Rows are the exact complete executed outer recipe commands in gap-free ordinal order: cpython has source_extract,configure,make,install; project-native has project_wheel,supervisor_compile,fp_native_compile,supervisor_link,fp_compile,fp_link. Each command has a separate stdout and stderr OS pipe drained to EOF without merging; counts are nonnegative, hashes bind exact bytes retained under object_store_rule, exit_code is exactly 0, and join occurs before publication. Child/subtool processes remain separately closed by build_exec_index, never create extra command-stream rows and inherit the owning outer row's two pipes. Thus no concatenation, delimiter, command-boundary or empty-stream ambiguity exists.",
          "schema": "temporac.native-build-command-stream-index.v5",
          "top_level_closed_keys": [
            "build_kind",
            "contract_sha256",
            "rows",
            "run_ordinal",
            "schema"
          ]
        },
        "depfile_rule": "Each of supervisor_compile,fp_native_compile,fp_compile uses exact -MD -MF -MT argv; -MMD forbidden. GCC writes one named depfile by command completion. Parse GNU make syntax as bytes under LC_ALL=C: backslash-LF sole continuation; backslash escapes space,tab,#,$,:,backslash; one unescaped target colon; duplicate targets,recipes,variables,wildcard,order-only,NUL,CR,malformed escape or extra rule fails. Target equals -MT. Normalize dependencies lexically after rejecting relative,dot/dotdot,symlink race/alias; strict byte-sort/deduplicate only after verifying emitted first-seen list. Set contains translation unit plus every user,CPython,libc,compiler-internal/system header because -MD includes system headers; each resolves one terminal input/toolchain/build-image row and reopened bytes/hash. There is no separate depfile-ledger artifact: original depfile is retained exactly once as kind=depfile build-output object, and every verifier deterministically recomputes the normalized projection from those raw bytes under this rule. Twin raw depfile byte equality plus deterministic parse proves projection equality. Missing header,unledgered generated header,-MMD,rewritten depfile or prefix mismatch fails.",
        "equality_projection": {
          "digest_rule": "equality_projection_sha256 is SHA-256 of the complete canonical projection bytes including LF and is absent from it.",
          "row_closed_keys": [
            "build_kind",
            "path",
            "role",
            "run0_build_output_sha256",
            "run1_build_output_sha256",
            "shared_bytes",
            "shared_kind",
            "shared_link_target_or_null",
            "shared_mode_octal",
            "shared_object_sha256"
          ],
          "rule": "The standalone projection has exact keys contract_sha256,rows,schema. Rows cover first all cpython then all project-native output role/path universes in strict build_kind,role,path byte order. Each pair resolves typed run0/run1 output rows; role/path/kind/mode/link/bytes/object bytes are identical and shared values repeat them, while stored row digests differ only because their normative preimages contain run_ordinal. For each build_kind, compare the two complete build-exec,build-input-pre,build-input-terminal,command-stream,generated-exec-image,temp-state indexes after replacing exactly top-level run_ordinal and every declared row run_ordinal with integer0. For generated-exec-image rows, recompute generated_exec_image_sha256 from the resulting normalized seven-key preimage before byte comparison. For each build-exec row with image_origin=prior-build-output, also replace image_provenance_sha256 by that recomputed normalized generated-exec-image row digest; tool-input provenance digests remain unchanged. Compare the two build-output indexes by the same ordinal replacement and recompute each build_output_sha256 from its resulting normalized nine-key preimage before byte comparison. These explicitly named digest rewrites are the complete ordinal-normalization transform; no other field or pointer changes. Within each run terminal input equals pre input after only phase replacement. Missing stage/build/ephemeral/project intermediate/stream/temp/object,unequal metadata/input/poststate or undeclared output fails.",
          "schema": "temporac.native-rebuild-equality-projection.v5"
        },
        "generated_exec_image_index": {
          "digest_rule": "generated_exec_image_index_sha256 is SHA-256 of the complete canonical index bytes including LF and is absent from the index. generated_exec_image_sha256 is SHA-256 of the standalone canonical seven-key projection build_kind,bytes,consumer_exec_ordinal,image_path,object_sha256,producer_exec_ordinal,run_ordinal plus LF and is stored as the eighth key.",
          "row_closed_keys": [
            "build_kind",
            "bytes",
            "consumer_exec_ordinal",
            "generated_exec_image_sha256",
            "image_path",
            "object_sha256",
            "producer_exec_ordinal",
            "run_ordinal"
          ],
          "row_rule": "The top-level build_kind/run_ordinal equal enclosing evidence and every row repeats both. project-native has empty rows because every executed image is tool-input. cpython rows are strict consumer ordinal and exactly one per build-exec prior-build-output. producer and consumer ordinals are nonnegative, producer strictly smaller, both resolve same-run exec rows; producer action created path. Immediately before consumer execve, controller opens regular nonsymlink O_RDONLY|O_CLOEXEC|O_NOFOLLOW, records positive bytes/object hash, retains bytes and closes/proves fd; exec path/hash/provenance equal row. Reuse of pathname creates distinct consumer rows and retains each version. Row digest validates exact seven-key preimage. Missing producer,future/self,path race,unretained ephemeral,mislabeled tool or plan absence fails.",
          "schema": "temporac.native-generated-exec-image-index.v5",
          "top_level_closed_keys": [
            "build_kind",
            "contract_sha256",
            "rows",
            "run_ordinal",
            "schema"
          ]
        },
        "index": {
          "digest_rule": "native_rebuild_evidence_index_sha256 is SHA-256 of the complete canonical index bytes including LF and is absent from it.",
          "row_closed_keys": [
            "build_exec_index_bytes",
            "build_exec_index_sha256",
            "build_input_pre_index_bytes",
            "build_input_pre_index_sha256",
            "build_input_terminal_index_bytes",
            "build_input_terminal_index_sha256",
            "build_kind",
            "build_output_index_bytes",
            "build_output_index_sha256",
            "command_stream_index_bytes",
            "command_stream_index_sha256",
            "generated_exec_image_index_bytes",
            "generated_exec_image_index_sha256",
            "ordinal",
            "temp_state_index_bytes",
            "temp_state_index_sha256"
          ],
          "row_rule": "Exactly four rows ordered (cpython,0),(cpython,1),(project-native,0),(project-native,1). Each nested index top-level build_kind/run_ordinal equals this row's build_kind/ordinal, and every nested row repeats it; build-output/generated-exec contain only that run. Every nested byte count is positive and digest resolves retained canonical JSON under object_store_rule. build-exec covers every process/join and reviewed plan projection; generated-exec resolves every prior-build-output. build_input_pre_index has phase=pre-run and terminal has phase=terminal; both are retrievable, and terminal equals pre byte-for-byte after replacing only phase as specified. output,temp-state,command-stream cover their exact closures. CPython rows retain configure/make/install,ephemeral images,complete twin build/stage exports; project-native starts only after CPython equality and mounts exact run0 stage. Twin equality covers both input phases and generated-exec after only run-ordinal normalization. Digest-only evidence,mutation,partial log,omitted empty stream,unclosed process,mixed-run index or learned output fails.",
          "schema": "temporac.native-rebuild-evidence-index.v5",
          "top_level_closed_keys": [
            "build_exec_plan_sha256",
            "contract_sha256",
            "equality_projection_sha256",
            "native_build_recipe_sha256",
            "rows",
            "schema"
          ]
        },
        "object_store_precondition": "The P00 creation in object_store_rule occurs before first non-authoritative preflight, because immutable toolchain ledger and later accepted build-exec plan must be retrievable there. At creation both subdirectories are empty; accepted toolchain ledger is first permitted index. The phrase no output can authorize its own build means no output may authorize its own production, recipe, producer or an earlier/same exec. It does not forbid a completed same-run CPython generated executable from serving a strictly later consumer when its exact path/hash/order was already constrained by the independently accepted plan and generated-exec-image evidence.",
        "object_store_root": "/var/lib/temporac/replay-v4/native-build-v5",
        "object_store_rule": "Before either authority-bearing CPython twin, a reviewed P00 receipt creates empty /var/lib/temporac/replay-v4/native-build-v5/{objects,indexes} owner0:0 mode0700 on one local filesystem. The nonroot build streams each completed object/log/index to the native build controller, which publishes owner0:0 mode0600 immutable files by the controller durability write algorithm under objects/<sha256>.bin or indexes/<sha256>.json, with content-addressed EEXIST only on exact byte/owner/mode equality. Evidence paths contain exactly lowercase digest basenames, are reopened O_RDONLY|O_CLOEXEC|O_NOFOLLOW with complete size/EOF/hash checks, and are retained through native-install, BASE, G5a, G5b and K7. Evidence index/equality rows are published only after all four runs join and all objects rehash. No recipe contains an output digest; no output can authorize its own build.",
        "schema": "temporac.native-rebuild-evidence-family.v5",
        "temp_state_index": {
          "digest_rule": "temp_state_index_sha256 is SHA-256 of the complete canonical index bytes including LF and is absent from the index.",
          "row_closed_keys": [
            "bytes",
            "gid",
            "kind",
            "mode_octal",
            "object_sha256_or_null",
            "path",
            "root_role",
            "run_ordinal",
            "symlink_target_or_null",
            "uid"
          ],
          "row_rule": "The index top-level build_kind/run_ordinal exactly equal the enclosing evidence row and every row repeats run_ordinal. It is the complete lstat roster, including the root row and empty directories, under exactly tmp,cache,home for cpython and project-native after the terminal command; project-native out is excluded because it is wholly in build-output, while cpython build/stage are wholly in build-output. Rows are strict root_role,path byte order, paths are relative and nonescaping, and root_role is cache,home,or tmp. kind is regular,directory,or symlink with the same bytes/hash/target/uid/gid/mode rules as build_input_terminal_index; regular/symlink objects are retained, directory hash is null. Socket,device,fifo,hardlink alias,mount change, undeclared root, outside write or omission fails. The two runs of one build_kind must be byte-identical after replacing only top-level/row run_ordinal by zero; allocator padding or filesystem enumeration order never enters canonical bytes.",
          "schema": "temporac.native-build-temp-state-index.v5",
          "top_level_closed_keys": [
            "build_kind",
            "contract_sha256",
            "rows",
            "run_ordinal",
            "schema"
          ]
        }
      },
      "project_build_input_index": {
        "digest_rule": "project_build_input_index_sha256 is SHA-256 of the complete canonical index bytes including LF. The digest is external and absent from the index.",
        "required_rows": [
          {
            "bytes": 8316,
            "path": "README.md",
            "sha256": "cbd538dddb0f4b70774e674710f279877ecfae37abeff58a0b32c555583a8ee3"
          }
        ],
        "row_closed_keys": [
          "bytes",
          "path",
          "sha256"
        ],
        "rule": "The future artifact has exactly top_level_closed_keys. rows equals required_rows byte-for-byte and in strict ASCII path order; each row binds the exact repository regular nonsymlink bytes and is disjoint from the 30-row executable source allowlist. contract_sha256 is the accepted Round7 successor. README.md is a build-only metadata input required by the bound pyproject.toml readme declaration; it is copied into no installed path and authorizes no import, compile or exec event. Missing, extra, changed, mutable-alias or digest-only input fails before Hatchling starts.",
        "schema": "temporac.project-build-input-index.v5",
        "top_level_closed_keys": [
          "contract_sha256",
          "rows",
          "schema"
        ]
      },
      "project_wheel_filename": "pams_rac_reproduction-0.1.0-py3-none-any.whl",
      "rebuild_equality": "After both CPython runs and their complete native-rebuild equality validate, create two new nonoverlapping project-native instances of the same immutable build image. In each, create /build/temporac root0:0 mode0755 and nonwritable to uid65532; before dropping privilege precreate exact root-owned mountpoints /build/temporac/src,pyproject.toml,README.md,cpython-stage plus initially empty out,tmp,cache owner65532:65532 mode0700 and home owner65532:65532 mode0500 under umask077. Materialize all 30 source-allowlist rows and the one disjoint README build-input in a verified root lower tree, permit no other repository member, then bind-mount only src and the two root files read-only with directories0555,regular0444,exact symlinks. Bind-mount exact validated run0 CPython stage export read-only at cpython-stage. Parent remains unmounted/nonwritable to uid65532, so only out,tmp,cache are writable and home remains nonwritable; /opt/temporac/replay-v4 is absent. Execute as uid=gid=65532, no groups/capabilities, no_new_privs=1, then run project_wheel,supervisor_compile,fp_native_compile,supervisor_link,fp_compile,fp_link in exact order with identical paths/env/argv. The -P -s Hatchling call honors PYTHONHASHSEED=0,PYTHONNOUSERSITE=1,PYTHONDONTWRITEBYTECODE=1. CPython headers/libpython resolve only from staged sysroot; non-CPython system/compiler headers,crt objects and libraries resolve only from exact toolchain/build-image ledger; ELF RPATH names future final /opt runtime. Every executable/input resolves through accepted exec/tool/source/build-input ledgers; temp/cache writes stay under TMPDIR/XDG, no undeclared writable/current-tree path exists. After six commands reopen/rehash all source,README,pyproject,CPython-stage/tool inputs and require exact pre-run byte/mode/link/mount equality. Require complete project-out bytes/hashes, ELF/ZIP projections, all three retained kind=depfile raw build-output objects plus deterministic resolver validation/projection, complete tmp/cache/home index and separate per-command stdout/stderr/exit evidence byte-identical across twins after only run-ordinal normalization. Native FP objects/depfiles are retained intermediate rows and never installed separately. Only run0 project wheel,supervisor and extension outputs enter native-install; any nondeterminism,undeclared input,mutation,metadata drift or run-specific path fails.",
      "recipe_artifact": {
        "closed_keys": [
          "contract_sha256",
          "normative_spec_index_sha256",
          "project_build_input_index_sha256",
          "schema",
          "source_allowlist_sha256",
          "toolchain_ledger_sha256"
        ],
        "digest_rule": "native_build_recipe_sha256 is SHA-256 of the complete canonical six-key recipe bytes including LF. The digest is external and absent from the recipe.",
        "normative_spec_index": {
          "closed_keys": [
            "contract_sha256",
            "rows",
            "schema"
          ],
          "digest_rule": "normative_spec_index_sha256 is SHA-256 of the complete canonical index bytes including LF and is absent from that index.",
          "row_closed_keys": [
            "json_pointer",
            "value_sha256"
          ],
          "row_order": [
            "/round7_native_wave2_closure/build_and_deploy/build_environment",
            "/round7_native_wave2_closure/build_and_deploy/commands",
            "/round7_native_wave2_closure/build_and_deploy/cpython",
            "/round7_native_wave2_closure/build_and_deploy/deployment",
            "/round7_native_wave2_closure/build_and_deploy/elf_acceptance",
            "/round7_native_wave2_closure/build_and_deploy/forbidden",
            "/round7_native_wave2_closure/build_and_deploy/fp_native_abi",
            "/round7_native_wave2_closure/build_and_deploy/future_populated_fields",
            "/round7_native_wave2_closure/build_and_deploy/hatchling",
            "/round7_native_wave2_closure/build_and_deploy/install_associations",
            "/round7_native_wave2_closure/build_and_deploy/native_install_index",
            "/round7_native_wave2_closure/build_and_deploy/native_rebuild_evidence",
            "/round7_native_wave2_closure/build_and_deploy/project_build_input_index",
            "/round7_native_wave2_closure/build_and_deploy/project_wheel_filename",
            "/round7_native_wave2_closure/build_and_deploy/rebuild_equality",
            "/round7_native_wave2_closure/build_and_deploy/source_purpose",
            "/round7_native_wave2_closure/build_and_deploy/toolchain"
          ],
          "row_rule": "Resolve each pointer in the exact accepted timestamped Amendment005 JSON and hash the complete canonical standalone value plus LF. Rows are exactly row_order. contract_sha256 is the accepted Round7 successor. Missing, extra, mutable-alias, self-pointer or alternate value encoding fails.",
          "schema": "temporac.native-build-normative-spec-index.v5"
        },
        "rule": "contract_sha256 is the accepted Round7 successor; normative_spec_index_sha256 resolves the exact static build/deploy choices; source_allowlist_sha256 resolves every executable project/build source byte including the three future members; project_build_input_index_sha256 resolves the exact separate build-only README.md bytes required by pyproject.toml; toolchain_ledger_sha256 resolves the complete exact future-populated tool/input ledger. All five digests must be reviewed before any build. The recipe contains no output digest, bundle digest, environment, launch or own digest.",
        "schema": "temporac.native-build-recipe.v5"
      },
      "recipe_schema": "temporac.native-build-recipe.v5",
      "source_purpose": {
        "src/pams/temporac/_fp_control.c": "Only the exact two-mode floating-point native ABI plus CPython extension wrappers, probe and readback. Its native object is linked into the supervisor and its extension object into _fp_control.so; it contains no teacher, metric, certificate, target or evaluator computation.",
        "src/pams/temporac/_origin_supervisor.c": "The sole controller and embedded-child executable source: request parsing, durable storage, fork/exec, FD/memfd transport, calls to the linked exact floating-point ABI, CPython initialization, native audit hook, framing, wait/join and fixed-error mapping. It contains no duplicate floating-point implementation.",
        "src/pams/temporac/protocol_entry.py": "The future byte-only dispatch ABI and exact owner table; it contains no native transport, build discovery or scientific-method change."
      },
      "toolchain": {
        "compiler_absolute_path": "/opt/temporac/toolchain/bin/x86_64-linux-gnu-gcc",
        "compiler_version": "GCC 13.2.0",
        "linker_version": "GNU ld 2.41",
        "target": "x86_64-linux-gnu",
        "target_isa": "x86-64-v2",
        "tool_ledger": {
          "build_exec_index": {
            "digest_rule": "build_exec_index_sha256 is SHA-256 of the complete canonical index bytes including LF. The digest is external and absent from the index.",
            "graph_and_resolver_rule": "Rows are gap-free exact process-creation ordinal order. For build_kind=cpython there are exactly four null-parent roots, in execution order source_extract,configure,make,install. Their stage tokens are respectively those four literals; source_extract argv equals commands.source_extract and cwd=/build/temporac, while configure/make/install argv equal the same run_ordinal row's configure_argv/make_argv/install_argv in cpython.build_runs and cwd=/build/temporac/cpython-build. For build_kind=project-native there are exactly six null-parent roots, in execution order project_wheel,supervisor_compile,fp_native_compile,supervisor_link,fp_compile,fp_link; each stage equals that literal, argv equals the same-named commands array and cwd=/build/temporac. Every nonroot has one strictly smaller parent_ordinal_or_null, inherits exactly the unique root's stage token, and its parent chain terminates at that root; orphan,cycle,cross-root,cross-command or invented stage fails. The reviewed plan rows obey the identical root/parent/stage/argv/cwd rule after their declared projection. Before each authority run the controller reopens and validates the already accepted native-build recipe, its toolchain ledger and reviewed plan. At top level, build_exec_plan_sha256 equals SHA-256 of those exact retained temporac.native-build-exec-plan.v5 bytes; recipe_normative_spec_index_sha256 equals normative_spec_index_sha256 inside that preexisting recipe; build_image_sha256 equals sha256 of the unique role=build-image row in that recipe's exact toolchain ledger. The later enclosing native-rebuild-evidence top-level native_build_recipe_sha256 and build_exec_plan_sha256 bind those same earlier inputs and never supply a value backward in time. For each image_origin=tool-input row, image_path and image_sha256 equal the unique executable tool-ledger row's absolute path_or_image_reference and sha256, and image_provenance_sha256 equals that row's tool_row_sha256. For each image_origin=prior-build-output row, image_path,image_sha256 and image_provenance_sha256 equal the same-run generated-exec-image row's image_path,object_sha256 and generated_exec_image_sha256 whose consumer_exec_ordinal equals this row ordinal. No arbitrary digest, alternate ledger, cross-run resolver or unresolved top-level value is accepted.",
            "row_closed_keys": [
              "argv",
              "cwd",
              "environment",
              "environment_sha256",
              "exit_code",
              "image_origin",
              "image_path",
              "image_provenance_sha256",
              "image_sha256",
              "ordinal",
              "parent_ordinal_or_null",
              "run_ordinal",
              "stage"
            ],
            "row_rule": "Top-level build_kind/run_ordinal identify one cpython or project-native run and every row repeats ordinal. Rows are exact process creation order governed by graph_and_resolver_rule. argv is a nonempty exact array; cwd is absolute and obeys cwd_rule; exit_code=0; image_path is regular nonsymlink and image_sha256 hashes the bytes opened by execve. environment is the complete actual execve envp as an ordered JSON string array: each element is strict UTF-8 KEY=VALUE, key matches ASCII [A-Za-z_][A-Za-z0-9_]*, keys are unique, no NUL occurs and order is preserved. environment_sha256 hashes the standalone canonical array plus LF. Every outer recipe-command row environment equals build_environment.envp byte-for-byte and in order. Child/subtool rows may differ only by their exact array frozen in the reviewed plan; no secret,clock,random,PID-dependent,host fallback or unreviewed variable is permitted, and authority rows match the plan. Non-env cwd/uid/gid/groups/capabilities/network/no_new_privs/umask remain bound by the build_environment normative digest/top recipe and are not packed into the environment hash. image_origin is exactly tool-input or prior-build-output and all image/provenance fields resolve under graph_and_resolver_rule. Shell builtins create no row but the containing image,script and input remain exact. Failed,extra,reordered,mixed-run,PATH-substituted or unledgered exec fails.",
            "schema": "temporac.native-build-exec-index.v5",
            "top_level_closed_keys": [
              "build_exec_plan_sha256",
              "build_image_sha256",
              "build_kind",
              "recipe_normative_spec_index_sha256",
              "rows",
              "run_ordinal",
              "schema"
            ]
          },
          "build_exec_plan": {
            "digest_rule": "build_exec_plan_sha256 is SHA-256 of the complete canonical plan bytes including LF and is absent from the plan. After independent acceptance and before any authority-bearing run, the native build controller publishes those exact bytes at native-build-v5/indexes/<build_exec_plan_sha256>.json under object_store_rule; every later reference reopens, EOF-checks, rehashes and byte-identically reserializes that retained object.",
            "preflight_exec_index": {
              "digest_rule": "preflight_build_exec_index_sha256 is SHA-256 of the complete canonical index bytes including LF and is external to that index. Exact bytes are published and retained under native_rebuild_evidence.object_store_rule before the preflight pipeline or plan is accepted.",
              "row_closed_keys": [
                "argv",
                "cwd",
                "environment",
                "environment_sha256",
                "exit_code",
                "image_origin",
                "image_path",
                "image_provenance_sha256",
                "image_sha256",
                "ordinal",
                "parent_ordinal_or_null",
                "run_ordinal",
                "stage"
              ],
              "rule": "This is the non-authorizing preflight analogue of temporac.native-build-exec-index.v5 and is captured before any plan or authority native-rebuild evidence exists. Top-level and row build_kind/run_ordinal, the exact four-or-six process-root sets, parent-chain/stage/argv/cwd clauses, complete environment and exit_code=0 obey the row and graph rules of the authority schema; its plan-dependent and later-native-rebuild resolver clauses are expressly inapplicable. Before each preflight, the controller reopens the already independently reviewed frozen native-build recipe and its toolchain ledger; recipe_normative_spec_index_sha256 equals that recipe's exact field and build_image_sha256 equals the unique role=build-image tool row sha256. A tool-input image_provenance_sha256 is the unique same-ledger tool_row_sha256; a prior-build-output value is the same-preflight-run generated_exec_image_sha256 with matching consumer ordinal. The later enclosing preflight-pipeline top-level native_build_recipe_sha256 binds those same previously existing recipe bytes and its row binds this exact preflight index digest; neither relationship supplies a value backward in time. Any reference to authority native-rebuild evidence, mixed authority bytes, plan backfill, alternate recipe/tool ledger, unresolved generated image, host fallback or post-capture rewrite fails.",
              "schema": "temporac.native-build-preflight-exec-index.v5",
              "top_level_closed_keys": [
                "build_image_sha256",
                "build_kind",
                "recipe_normative_spec_index_sha256",
                "rows",
                "run_ordinal",
                "schema"
              ]
            },
            "preflight_pipeline": {
              "digest_rule": "preflight_pipeline_sha256 is SHA-256 of the complete canonical pipeline bytes including LF and is absent from it. The pipeline and every nested index are immutable retrievable objects under native_rebuild_evidence.object_store_rule; the later accepted plan contains only this external digest plus its run-independent row projection.",
              "row_closed_keys": [
                "build_input_pre_index_sha256",
                "build_input_terminal_index_sha256",
                "build_kind",
                "build_output_index_sha256",
                "command_stream_index_sha256",
                "generated_exec_image_index_sha256",
                "preflight_build_exec_index_sha256",
                "run_ordinal",
                "temp_state_index_sha256"
              ],
              "rule": "contract_sha256 is the accepted Round7 successor and native_build_recipe_sha256 hashes the exact independently reviewed frozen recipe; neither may be caller-selected or learned from a preflight. Rows are exactly (cpython,0),(cpython,1),(project-native,0),(project-native,1). First execute only the two fresh CPython preflights under that recipe; in each row preflight_build_exec_index_sha256 resolves only a retained temporac.native-build-preflight-exec-index.v5 object, never an authority temporac.native-build-exec-index.v5 object, and the other fields resolve complete input-pre/input-terminal/output/command-stream/generated-exec/temp-state indexes using the same closed schemas and byte-retention rules as authority evidence. Validate their complete twin equality under native_rebuild_evidence.equality_projection, including normalized typed row digests. selected_cpython_stage_build_output_index_sha256 then equals only the CPython preflight run0 build-output index digest; its complete cpython-stage-export projection is byte-identical to run1 and is mounted read-only at /build/temporac/cpython-stage for each later project-native preflight, with a complete bijection to that preflight's input-pre and terminal cpython-stage rows. Only after this equality do the two project-native preflights run; their complete nested indexes likewise validate twin equality. For each build kind, projecting either preflight exec index by deleting exit_code,run_ordinal,image_provenance_sha256 and injecting enclosing build_kind yields exactly the same ordered plan rows. No preflight output, selected stage or nested digest may enter native-install, substitute for an authority output, grant experiment authority or modify science; it survives only as immutable evidence transitively resolved by the reviewed plan and is retained through K7. Missing output object, CPython-stage source ambiguity, host/authority-stage use, cross-run mount, unequal twin, plan backedge or discarded evidence fails.",
              "schema": "temporac.native-build-preflight-pipeline.v5",
              "top_level_closed_keys": [
                "contract_sha256",
                "native_build_recipe_sha256",
                "rows",
                "schema",
                "selected_cpython_stage_build_output_index_sha256"
              ]
            },
            "row_closed_keys": [
              "argv",
              "build_kind",
              "cwd",
              "environment",
              "environment_sha256",
              "image_origin",
              "image_path",
              "image_sha256",
              "ordinal",
              "parent_ordinal_or_null",
              "stage"
            ],
            "rule": "The future artifact has exactly top_level_closed_keys. preflight_pipeline_sha256 resolves the complete validated four-row non-authorizing pipeline above. Rows are cpython then project-native and within kind gap-free process order; for each kind they byte-equal the unique run-independent projection reproduced by both corresponding preflight exec indexes. Rows contain no run ordinal,exit code,image-provenance row digest,authority output digest or authority-run value. Each embeds the exact actual ordered environment array and its standalone-array digest; outer rows equal build_environment.envp, while subtool differences are fully explicit/reviewed and deterministic with no secret,clock,random,PID or host value. image_origin is tool-input or prior-build-output; the plan commits exact image path/object hash, only tool input exists pre-build, and prior output is restricted to CPython generated tools equal across both CPython preflights. Independent review accepts the complete pipeline and this one plan before any authority-bearing run. Each authority exec index binds the plan; remove only exit_code,run_ordinal,image_provenance_sha256, inject enclosing build_kind, canonicalize and require row byte equality. No other field changes and every exit code is zero. Preflight artifacts are transitive plan evidence only and never native-install or scientific outputs.",
            "schema": "temporac.native-build-exec-plan.v5",
            "top_level_closed_keys": [
              "contract_sha256",
              "preflight_pipeline_sha256",
              "rows",
              "schema"
            ]
          },
          "digest_rule": "toolchain_ledger_sha256 is SHA-256 of the complete canonical native-toolchain-ledger bytes including LF and is absent from the ledger. Before any preflight it is independently accepted and published owner0:0 mode0600 at /var/lib/temporac/replay-v4/native-build-v5/indexes/<toolchain_ledger_sha256>.json; every recipe/plan/run reference reopens O_RDONLY|O_CLOEXEC|O_NOFOLLOW, validates regular file/EOF/bytes/hash and byte-identical canonical reserialization.",
          "population_rule": "The native-build recipe and pure immutable toolchain ledger are accepted before preflight and contain no run-output digest. Run the two non-authoritative CPython clean-image preflights first, retain their complete typed indexes and validate full twin output/evidence equality. Mount only the validated run0 preflight stage projection read-only into two later non-authoritative project-native preflights; retain and compare their complete indexes. Publish the exact four-row preflight pipeline, independently review its two-kind run-independent exec projection and accept one plan. Only then may the four authority-bearing runs start, each matching the reviewed plan row-for-row while retaining its own exit/run evidence and all declared outputs; authority project-native runs still mount only authority CPython run0 after authority CPython twin equality. These provenance validations are not experiment jobs and add no GPU hours,data access or claim. Preflight bytes cannot enter native-install or authorize scientific execution and are retained only as exact plan evidence transitively resolved by native-rebuild evidence.",
          "required_role_order": [
            "build-image",
            "bash",
            "bsdtar",
            "libarchive",
            "liblzma",
            "gcc-driver",
            "cc1",
            "assembler",
            "linker",
            "archiver",
            "make",
            "libc-headers",
            "linux-headers",
            "gcc-headers",
            "crt-objects",
            "libgcc",
            "libstdcxx",
            "libc",
            "libm",
            "libdl",
            "libpthread",
            "dynamic-loader",
            "hatchling-wheel",
            "hatchling-build-environment"
          ],
          "row_closed_keys": [
            "bytes",
            "path_or_image_reference",
            "role",
            "sha256",
            "tool_row_sha256",
            "version"
          ],
          "rule": "Rows begin with exactly one nonnull row for every required_role_order token in that listed order. Any additional immutable pre-build executable is role=exec-image and follows in strict UTF-8 absolute path_or_image_reference order. role/path pairs and absolute paths are unique; a required-role row that itself names an executable directly serves as that image and is never duplicated as exec-image. Each row's tool_row_sha256 is SHA-256 of the standalone canonical five-key projection bytes,path_or_image_reference,role,sha256,version plus LF. Files bind exact absolute regular nonsymlink bytes/hashes; directory-like header/crt/build-image/Hatchling roles bind complete retained ASCII-path ordered member-index bytes/hash and use a fixed image-reference token declared by role. Every external executable permitted by bash/make/GCC/binutils/Hatchling/CPython build orchestration has exactly one row; CPython executables generated within the same run are deliberately absent and use build-exec image_origin=prior-build-output. The stated GCC/ld versions and target reproduce from exact binaries. This ledger contains no preflight/authoritative exec or output digest. Missing/duplicate/unindexed extra, host fallback, generated-output backedge or digest-only input fails.",
          "schema": "temporac.native-toolchain-ledger.v5",
          "top_level_closed_keys": [
            "contract_sha256",
            "rows",
            "schema"
          ]
        }
      }
    },
    "code_and_install_closure": {
      "association_preimage": {
        "closed_keys": [
          "bytes",
          "distribution",
          "generated_member_row_sha256_or_null",
          "installed_path",
          "kind",
          "member_tag",
          "normalized_member_path_or_null",
          "repository_path_or_null",
          "repository_sha256_or_null",
          "repository_tag",
          "schema",
          "sha256",
          "version",
          "wheel_member_path_or_null",
          "wheel_member_sha256_or_null",
          "wheel_sha256"
        ],
        "hash": "association_sha256 is SHA-256 of this exact standalone canonical JSON object including LF; the installed index row repeats the fifteen non-schema values and adds only association_sha256.",
        "schema": "temporac.installed-member-association.v4"
      },
      "completion_index": {
        "digest_rule": "executable_origin_trace_completion_index_sha256 is SHA-256 of the complete canonical index bytes including LF and is external to the index. The exact bytes remain retrievable through G5a,G5b and K7.",
        "row_closed_keys": [
          "entrypoint_owner",
          "launch_instance_bytes",
          "launch_instance_sha256",
          "trace_bytes",
          "trace_sha256"
        ],
        "rule": "The G5a-bound index has exactly top_level_closed_keys and exactly 54 rows: the first 54 dispatch owner strings, excluding G5A-COMMIT,G5B-JOIN,K7-FINAL, each once in strict printable-ASCII entrypoint_owner order. launch_instance_bytes and trace_bytes are positive exact canonical byte counts including LF and their digests hash those complete retained bytes. Each launch parses as temporac.native-launch-instance.v5 and resolves one exact full-runtime launch-evidence row; each trace parses as temporac.executable-origin-trace.v5, repeats the same launch digest, contract, source/installed/runtime/FINAL environment lineage, controller request, stream/join/mapping evidence and status PASS. The request-keyed durability attestation must resolve that same final_trace_sha256 and launch evidence before the row is eligible. Top contract_sha256 is the accepted Round7 successor, environment_sha256 hashes the one exact final CPU environment and source_allowlist_sha256 resolves the BASE allowlist. The three post-G5a owners are finalized only as surrounding controller trace evidence after their calls return and never enter or mutate this pre-G5a index, receipt root, capability or result. Missing/extra/duplicate owner, v4 trace substitution, digest-only launch, cross-request attestation, post-G5a backfill or trace-to-environment feedback fails.",
        "schema": "temporac.executable-origin-trace-completion-index.v5",
        "top_level_closed_keys": [
          "contract_sha256",
          "environment_sha256",
          "rows",
          "schema",
          "source_allowlist_sha256"
        ]
      },
      "installed_ledger": "The preserved installed-distribution-member-index is complete, ASCII-path ordered and duplicate-free for exactly the noneditable project wheel and every third-party wheel. Each accepted row is wheel-member and binds distribution, version, normalized path, kind, exact installed bytes/count/hash, wheel member path/hash and repository path/hash or explicit no-repository-member through the standalone 16-key v4 association preimage. CPython members are instead complete in cpython-runtime-member-index; supervisor and _fp_control are instead native-build-output rows in native-install-index plus ELF index; none is falsely encoded by the wheel-only v4 association. No glob or dynamic fallback exists.",
      "normalization": "Use the preserved normalize_rel_posix_ascii algorithm: printable strict ASCII, NFC unchanged, slash only, no NUL/backslash/colon/absolute/trailing/repeated slash/empty/dot/dotdot segment, no case fold, percent decode, realpath or separator rewrite. Every present/null discriminator and installed/wheel/repository/origin kind is explicit in the canonical association row.",
      "origin_preimage": "Every trace origin is one closed typed canonical JSON object with a kind-specific exact key set. origin_sha256 is SHA-256 of those standalone bytes including LF; it contains distribution/version/member/install/repository/native/generated identity as applicable. Alternate tagged binary, delimiter encoding, renamed field, omitted null, path alias, member swap or digest-only substitute fails.",
      "project_equality": "For every project-installed row, installed bytes equal the corresponding project-wheel member and repository association byte-for-byte. For every imported/executed project row, those same bytes additionally equal one exact precommitted source-allowlist row; non-allowlisted installed project members have exact zero executable/import events. protocol_entry is loaded only from this noneditable installed wheel under the isolated module_search_paths.",
      "runtime_entry": "There is no python -I -m bootstrap. execve enters the exact native supervisor ELF; the child installs its audit hook, validates transport, configures isolated embedded CPython with the preserved 10+64 field profile and exact module_search_paths, then imports protocol_entry only after the pre-dispatch import/trusted-generated prefix.",
      "source_allowlist": {
        "digest_rule": "source_allowlist_sha256 is SHA-256 of the complete canonical temporac.source-byte-allowlist.v5 bytes including LF. The digest is external and absent from the index.",
        "required_repository_paths_ascii_order": [
          "pyproject.toml",
          "src/pams/__init__.py",
          "src/pams/temporac/__init__.py",
          "src/pams/temporac/_fp_control.c",
          "src/pams/temporac/_origin_supervisor.c",
          "src/pams/temporac/certify.py",
          "src/pams/temporac/contract.py",
          "src/pams/temporac/cue.py",
          "src/pams/temporac/decode.py",
          "src/pams/temporac/evaluator.py",
          "src/pams/temporac/feature_io.py",
          "src/pams/temporac/fixtures.py",
          "src/pams/temporac/gates.py",
          "src/pams/temporac/hashio.py",
          "src/pams/temporac/metrics.py",
          "src/pams/temporac/nola.py",
          "src/pams/temporac/objective.py",
          "src/pams/temporac/prediction.py",
          "src/pams/temporac/preprocess.py",
          "src/pams/temporac/protocol_entry.py",
          "src/pams/temporac/quadrature.py",
          "src/pams/temporac/receipts.py",
          "src/pams/temporac/response.py",
          "src/pams/temporac/runtime.py",
          "src/pams/temporac/teacher.py",
          "src/pams/temporac/training.py",
          "src/pams/temporac/trusted_packer.py",
          "src/pams/temporac/types.py",
          "src/pams/temporac/x0.py",
          "src/pams/types.py"
        ],
        "row_closed_keys": [
          "bytes",
          "path",
          "roles",
          "sha256"
        ],
        "row_rule": "rows occur exactly in required_repository_paths_ascii_order. bytes is a positive integer, sha256 hashes the exact complete repository file bytes, path repeats the listed ASCII path, and roles is a nonempty ASCII-sorted unique subset of build-input,project-metadata,python-import. pyproject has [build-input,project-metadata]; the two C sources have [build-input]; every Python source has [build-input,python-import] because the noneditable wheel build consumes it. No other role token exists.",
        "rule": "The list is exact, complete, ASCII byte ordered and no-glob. A separately authorized future implementation must add the three presently absent Round7 sources and obtain an implementation review before any P00/P06 build; their present absence is SPECIFIED_NOT_IMPLEMENTED and blocks execution, not this specification review. Every imported/executed repository byte resolves to exactly one row and every row resolves to an installed/build association; no current-tree fallback or unlisted dynamic open exists.",
        "schema": "temporac.source-byte-allowlist.v5",
        "top_level_closed_keys": [
          "contract_sha256",
          "rows",
          "schema"
        ]
      },
      "trace_role": "The post-run executable-origin trace validates every imported/executed/native origin against the precommitted installed/source/ELF ledgers. It is evidence only and never feeds back into those ledgers, BASE, FINAL, environment, launch or capability."
    },
    "controller": {
      "argv": {
        "child": [
          "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
          "--embedded-cpython-v4"
        ],
        "controller": [
          "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
          "--controller-v4"
        ]
      },
      "child_fd_actions": {
        "controller_output_preservation": "After the eleven-key request has been fully read, canonicalized, hashed and durably reopened, require every controller fd from 3 through RLIMIT-1 closed. Duplicate original FD1 once with fcntl(F_DUPFD_CLOEXEC,32) and require exactly 32; then duplicate original FD2 with fcntl(F_DUPFD_CLOEXEC,33) and require exactly 33. Close controller FD0, FD1 and FD2 with EBADF readback. FD32 is the sole controller result-pipe OFD and FD33 the sole controller stderr /dev/null OFD retained across child launches; neither is inherited through exec or named by a launch.",
        "per_child_creation_order": [
          "Open /dev/null three independent times with O_RDONLY|O_CLOEXEC|O_NOFOLLOW and require the returned descriptors are exactly 0,1,2 in order, character device 1:3, distinct OFDs and offset zero. These are low-fd reservations only, never launch rows: target FD0 installation atomically replaces reservation0, and later child stdout/stderr installation replaces reservations1/2. This step is repeated fresh after run0 cleanup before discovery run1.",
          "With only low reservations FD0/1/2 and controller-only FD32/33, plus no open durable-directory/file descriptor, construct the mode-specific sealed operational inputs in numeric target-fd order by operational_memfd_construction and complete each INSTALL_TARGET in the controller process; target FD0 contains request.payload bytes, FD3/4/5 the exact FINAL roles/environment when full, and FD8 the request bundle.",
          "Open /dev/null twice independently with O_WRONLY|O_CLOEXEC|O_NOFOLLOW, promote them in open order by fcntl(F_DUPFD_CLOEXEC,40) requiring FD40 and fcntl(F_DUPFD_CLOEXEC,41) requiring FD41, and close/prove both raw descriptors. Prove distinct OFDs by F_SETFL-adding O_NONBLOCK only on FD40, requiring FD40 changed and FD41 remained exact O_WRONLY|O_LARGEFILE, then restore and reread FD40 exact before install. Install them by dup3 onto FD1 then FD2 with flags 0 replacing the reservations, close/prove FD40/41, and require the exact launch fd-row projections.",
          "Call pipe2(O_CLOEXEC) exactly once, promote the read end with fcntl(F_DUPFD_CLOEXEC,48) and require returned fd exactly 48, close/prove its raw source, install the write end by dup3 onto FD6 with flags 0, close/prove its raw source, and require one empty pipe with exactly the FD48 controller read peer and FD6 child write peer.",
          "Construct and persist the launch instance from the complete operational fd rows and sealed-input construction index, then construct/persist the FD7 carrier through its independent eight-stage algorithm. Immediately before fork require the only open descriptors are child targets for the exact mode plus controller-only FD32, FD33 and FD48; every durable root/file fd is closed and revalidated through its saved inode evidence.",
          "Call fork exactly once. In the child branch close FD32, FD33 and FD48 in that order, require EBADF, make no allocation or non-async-signal-safe call, and execve the exact supervisor with the fixed child argv/envp. In the parent branch close every child target fd in increasing order including FD6/FD7, retain only FD32, FD33 and FD48, drain FD48 to terminal EOF, close/prove it, waitpid the exact child, and construct that child's complete prelaunch/launch/stream/join/attestation rows in memory before preparing another discovery child or final result. For discover-twin the combined two-row indexes are durably published only after run1 joins; the durable nonretry attempt marker makes any intervening failure terminal and nonresumable."
        ],
        "restoration": "On the success path, after the one full child or both sequential discovery children have been durably joined, require FD1 and FD2 free; dup3(FD32,FD1,0), then dup3(FD33,FD2,0), close/prove FD32 and FD33, and require FD1 is the original controller result-pipe OFD while FD2 is the original independent write-only /dev/null OFD, both with descriptor flags zero and exact status. Controller FD0 and every other descriptor remain closed. Only then may the controller emit its one result frame on FD1 with no FD2 byte; after the complete frame it closes FD1 and FD2 and exits. On any failure after preservation, first close/prove every child target, pipe peer, transient and durable fd and wait any created child exactly once, then attempt the same 32-to-1 and 33-to-2 restoration before the sole error frame. If the original result OFD, FD32 or its 32-to-1 restoration is unusable, emit no bytes and exit 124. If only FD33 or 33-to-2 restoration is unusable while FD1 is valid, keep FD2 closed, select OUTPUT only if no earlier failure exists, emit the exact error on FD1, close FD1 and exit 123; no stderr byte exists. Otherwise restoration/cleanup never changes the already selected earlier failure stage.",
        "scratch_rule": "Targets 0..8, retained controller fds32/33, transient null fds40/41 and per-child read peer fd48 are the only fixed descriptors. Low reservations ensure every memfd writer satisfies the preserved fd>=3 construction rule; each raw syscall return must be the lowest currently free descriptor dictated by the verified table. All raw memfd/null/pipe descriptors are transient, must be promoted/installed exactly as specified, and are closed before fork; operational reader promotion minimum64 is preserved. Immediately before carrier construction, discovery has targets 0,1,2,6,8 plus 32,33,48 and therefore requires carrier writer_fd=3 and reader_fd=4; full has targets 0..6 except 7 plus 8 and 32,33,48 and therefore requires carrier writer_fd=7 and reader_fd=9. In both cases the writer closes before one exact dup3(reader_fd,7,0), so no reader-already-7 branch exists. A returned descriptor different from the value prescribed here or by the numbered action, collision, controller output/stderr OFD in a child, child target in the controller after fork, durable fd at fork, shared /dev/null OFD, second pipe, dup fallback, close-range ambiguity, retry or descriptor leak fails before exec evidence."
      },
      "durability": {
        "attempt_marker": {
          "closed_keys": [
            "controller_request_sha256",
            "created_before_child_count",
            "mode",
            "schema",
            "state"
          ],
          "rule": "Immediately after complete request validation/persistence and before any prelaunch construction or child creation, create attempts/<controller_request_sha256>.json with O_EXCL and no EEXIST acceptance. created_before_child_count=0, mode equals the request, state=CLAIMED and schema=temporac.native-controller-attempt-marker.v5. attempt_marker_sha256 hashes the complete canonical marker bytes including LF. It remains durable on success or failure; thus any valid request retry fails before child creation. An unparseable request has no digest/marker and grants no capability.",
          "schema": "temporac.native-controller-attempt-marker.v5"
        },
        "attestation_closed_keys": [
          "attempt_marker_sha256",
          "child_stream_index_sha256_or_null",
          "controller_request_sha256",
          "discovery_evidence_index_sha256_or_null",
          "final_trace_sha256_or_null",
          "launch_evidence_index_sha256_or_null",
          "mode",
          "process_join_index_sha256_or_null",
          "result_payload_sha256",
          "schema",
          "state",
          "status"
        ],
        "attestation_schema": "temporac.native-controller-durability-attestation.v5",
        "attestation_values": {
          "discover-twin": "state=DURABLE_REHASH_COMPLETE and status=PASS; attempt_marker_sha256 and child_stream_index_sha256_or_null, discovery_evidence_index_sha256_or_null, launch_evidence_index_sha256_or_null and process_join_index_sha256_or_null are all present exact lowercase digests; final_trace_sha256_or_null is null; result_payload_sha256 hashes the complete discovery controller result.",
          "full-runtime": "state=DURABLE_REHASH_COMPLETE and status=PASS; attempt_marker_sha256 and child_stream_index_sha256_or_null, final_trace_sha256_or_null, launch_evidence_index_sha256_or_null and process_join_index_sha256_or_null are all present exact lowercase digests; discovery_evidence_index_sha256_or_null is null because the already validated discovery digest is resolved through FINAL; result_payload_sha256 hashes the complete dispatch/controller inner result.",
          "rule": "mode is exactly discover-twin or full-runtime. Any other state/status token, null/present pattern, digest not resolving to the just-persisted object, cross-request object, missing launch evidence, or attestation created before reopen/rehash fails. Failure creates no PASS attestation."
        },
        "durable_paths": [
          "requests/<controller_request_sha256>.json",
          "attempts/<controller_request_sha256>.json",
          "prelaunch/<launch_instance_sha256>.json",
          "launch/<launch_evidence_index_sha256>.json",
          "streams/<child_stream_index_sha256>.bin",
          "joins/<process_join_index_sha256>.json",
          "traces/<final_trace_sha256>.json",
          "results/<result_payload_sha256>.bin",
          "attestations/<controller_request_sha256>.json"
        ],
        "eexist": "A state/attestation target that already exists is failure. A content-addressed immutable object may already exist only when reopen, fstat, exact bytes, digest, owner uid/gid and mode all equal; otherwise failure. No overwrite occurs.",
        "path_key_rule": "requests,launch,streams,joins,traces and results are keyed by the SHA-256 of their complete stored bytes. attempts and attestations are request-keyed state objects and prelaunch is keyed by the exact embedded launch digest although its seven-key wrapper has different bytes; those three are never called content-addressed and EEXIST always fails for state/attestation while prelaunch EEXIST requires exact wrapper equality. Angle-bracket tokens are substituted by exactly one 64-lowercase-hex digest with no delimiter or case alternative.",
        "prelaunch_record": {
          "closed_keys": [
            "carrier_construction_bytes_hex",
            "carrier_construction_sha256",
            "launch_instance_bytes_hex",
            "launch_instance_sha256",
            "operational_memfd_construction_index_bytes_hex",
            "operational_memfd_construction_index_sha256",
            "schema"
          ],
          "rule": "Each bytes_hex is lowercase complete canonical bytes including LF and hashes exactly. Launch resolves the same controller request/mode and the operational construction index resolves exactly its fd rows while excluding FD7. Carrier construction resolves exact FD7 evidence and contains the canonical carrier over these launch bytes. The complete seven-key prelaunch record is persisted at prelaunch/<launch_instance_sha256>.json and reopens byte-identically before fork; the path key is the embedded launch digest, not a digest of the wrapper, and EEXIST still requires complete wrapper equality.",
          "schema": "temporac.native-prelaunch-record.v5"
        },
        "publish_order": "Persist and rehash request, then atomically create/fsync/reopen the nonretry attempt marker, both before child creation. Persist each exact prelaunch record containing launch, operational-memfd construction and carrier-construction evidence before that fork. After all mode-required children reach terminal EOF and exact waitpid, persist the complete one-row/full or two-row/discover launch-evidence index, complete child-stream index and complete process-join index; assemble/persist trace when full and persist the exact mode-specific inner result payload; construct attestation without its own digest or enclosing controller-success envelope; fsync and reopen all objects; only then emit controller stdout success. Discovery run0's rows are completed in memory before run1 exists, while combined indexes are published only after run1; the already durable attempt marker forbids retry or resume after any intervening failure.",
        "retention": "Every published request, attempt marker, prelaunch record, launch evidence index, child-stream index, process-join index, final trace, inner result and durability attestation is immutable and remains readable at its exact durable path through G5a, G5b and K7 completion. Failed-attempt request/marker/prelaunch evidence is likewise retained for nonretry audit but grants no PASS authority. Deletion, garbage collection, path migration, mutable alias, digest-only replacement or early reuse fails the corresponding downstream resolver.",
        "root": "/var/lib/temporac/replay-v4/native-controller-v4",
        "root_mode_octal": "0700",
        "root_precondition": "A separately reviewed P00 deployment receipt proves that the root and exact child directories requests,attempts,prelaunch,launch,streams,joins,traces,results,attestations exist before the first controller request, are initially empty, regular directories on one local filesystem, owner 0:0, mode 0700, nonsymlink, non-mountpoint, and opened through one O_DIRECTORY|O_RDONLY|O_CLOEXEC|O_NOFOLLOW root fd. On every later request the controller revalidates the same root/directory identities, ownership/modes and mount. Existing entries are permitted only when they are immutable retained objects from prior request keys, each filename, schema, bytes/hash, owner/mode and cross-edge validates under durable_paths, path_key_rule, eexist and retention; no unrelated, malformed, mutable or unresolvable entry is permitted. For the current request, EEXIST handling applies only to the exact target path being created and follows eexist without treating valid retained objects under other request keys as a dirty root. The runtime installation under /opt is immutable and never contains mutable controller state.",
        "write_algorithm": "Open only one exact preopened child-directory fd beneath the verified root. The temporary basename is dot + final basename + .tmp. + unsigned decimal getpid() + . + unsigned decimal /proc/self/stat starttime_ticks, with no leading-zero alternative; openat it with O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW and mode 0600. Reject links/nonregulars; complete EINTR-safe write loop; fdatasync then fsync; close; renameat2 within that same directory with RENAME_NOREPLACE; fsync directory; reopen final basename O_RDONLY|O_CLOEXEC|O_NOFOLLOW; verify type/owner/mode/size/EOF/SHA; never chmod after publish, cross directories, retry a name, consult clock/random, or use path traversal."
      },
      "environment": {
        "cwd": "/",
        "envp": [
          "BLIS_NUM_THREADS=1",
          "LANG=C",
          "LC_ALL=C",
          "MKL_DYNAMIC=FALSE",
          "MKL_NUM_THREADS=1",
          "NUMEXPR_NUM_THREADS=1",
          "OMP_DYNAMIC=FALSE",
          "OMP_NUM_THREADS=1",
          "OPENBLAS_NUM_THREADS=1",
          "PYTHONHASHSEED=0",
          "TZ=UTC",
          "VECLIB_MAXIMUM_THREADS=1"
        ],
        "gid": 0,
        "uid": 0,
        "umask_octal": "077"
      },
      "external_fd_contract": [
        "fd0 is one sealed read-only request memfd at offset zero",
        "fd1 is the sole framed controller-result write pipe",
        "fd2 is an independently opened write-only /dev/null",
        "all other descriptors are closed before controller request parsing"
      ],
      "fork_exec": "Controller initializes no CPython. child_fd_actions installs every target in the controller before fork. For each child it uses fork exactly once; between fork and execve the child performs only close(32), close(33), close(48), their exact EBADF fcntl checks, and execve on the exact absolute supervisor path. A close/readback failure performs immediate _exit(125); an execve return performs immediate _exit(126). Neither path writes or allocates, and the controller maps 125 to FORK and 126 to EXEC and synthesizes the exact failure frame. vfork, posix_spawn, post-fork dup, allocation, shell, PATH lookup, inherited extra descriptors and retry are forbidden.",
      "modes": {
        "discover-twin": "One controller request launches exactly two fresh children sequentially, run_id/process_ordinal 0 then 1; run0 is fully drained and joined and its complete rows are materialized in controller memory before run1 fork. The exact combined two-row indexes are durably published only after run1 joins; any failure after the already durable attempt marker is terminal and cannot be retried or resumed.",
        "full-runtime": "One controller request launches exactly one child and returns only after the full child stream, EOF, waitpid, final trace, result and durability attestation validate."
      },
      "owner": "The same exact _origin_supervisor.c source and installed ELF own controller and child modes. No Python wrapper, second controller image, unbound service, daemon, RPC, shell or editable entrypoint exists.",
      "request": {
        "closed_keys": [
          "environment_bytes_hex_or_null",
          "environment_bytes_or_null",
          "environment_sha256_or_null",
          "ledger_bundle_bytes",
          "ledger_bundle_bytes_hex",
          "ledger_bundle_sha256",
          "mode",
          "payload_bytes",
          "payload_bytes_hex",
          "payload_sha256",
          "schema"
        ],
        "rules": [
          "ledger_bundle_bytes and payload_bytes are positive JSON integer byte counts; environment_bytes_or_null is null or positive. Each present *_bytes_hex is lowercase even-length hexadecimal of exactly that many complete bytes and the paired digest is SHA-256 of those bytes.",
          "Discovery uses mode discover-twin, BASE bundle, the exact discovery-controller-input wrapper as the sole payload, and all three environment fields null.",
          "Full uses mode full-runtime, FINAL bundle, exact temporac.invocation.v4 bytes as the sole payload, and all three environment fields present; parsed environment must bind that FINAL bundle digest.",
          "The complete canonical eleven-key request including LF and every hex expansion is at most controller_request_max_bytes=68719476735; this is checked before FD0 allocation/read. Its embedded ledger bundle remains at most 17179869184 and each embedded protocol payload/environment also passes its own lower-layer cap. No raw-byte count is substituted for complete request size.",
          "The request has no self digest. controller_request_sha256 is computed externally over its complete canonical JSON bytes including LF after validation."
        ],
        "schema": "temporac.native-controller-request.v5"
      },
      "resource_limits": {
        "rlimit_nofile_cur": 256,
        "rlimit_nofile_max": 256,
        "rule": "At native controller physical entry, before reading FD0 or opening any descriptor, call getrlimit(RLIMIT_NOFILE) and require finite initial rlim_cur in [256,1048576] and initial rlim_max at least 256; initial rlim_max may be finite or RLIM_INFINITY. Scan every integer 0..initial_rlim_cur-1 by fcntl(F_GETFD), retrying only EINTR, and require exactly FD0/1/2 open with the external contract and every other integer EBADF. Then call setrlimit once with cur=max=256, read back exact 256/256, and use that unchanged pair for every controller and child scan. This bounds the complete 0..255 descriptor universe while permitting fixed FD32/33/40/41/48 and simultaneous operational writer64/reader65. Any initially open extra, infinite initial rlim_cur, other rlim_cur bound, initial rlim_max below256, set/readback drift or later rlimit mutation fails before request parsing.",
        "schema": "temporac.native-resource-limit-profile.v5"
      },
      "result_stream": {
        "error_payload": "the exact three-key temporac.native-failure.v5 object",
        "exit_rules": "The sole frame always has sequence zero. Success type 0x7e then EOF requires controller exit 0. A validated child 0x7f failure; controller-synthesized EXEC failure; signal or nonprotocol child exit; or any malformed,truncated,short,early-EOF or broken FD6 child stream selected as STREAM_PARSE emits type 0x7f then EOF and controller exit 122. A controller request, operational construction, launch/PRELAUNCH, FORK, waitpid syscall or join-evidence validation, persistence, durability or controller-side serialization failure emits type 0x7f then EOF and exit 123. The sole exception to the normal child-origin 122 rule is the exact restoration case where FD1/FD32 remain usable but only FD33 or its 33-to-2 restoration fails: the already selected child failure stage/payload is retained, the frame is emitted on restored FD1, and controller exit is 123; if no earlier failure existed, that same condition selects OUTPUT and exit123. FD1/FD32/output-write failure follows output_failure_rule and exit124 always invalidates any zero/partial/complete observed frame. These cause classes are mutually exclusive; no generic parse/broken-stream category also selects exit123. Any signal on the controller, other code, missing EOF or success before durability fails.",
        "framing": "ASCII temporac.native-controller-stream.v5, NUL, then exactly one standard v5 frame and EOF; type 0x7e is success, 0x7f is error; any trailing byte, missing EOF, second frame or stderr byte fails.",
        "inner_result_rule": {
          "discover-twin": "The inner result bytes are exactly the complete canonical temporac.generated-source-discovery-evidence-index.v5 bytes including LF. payload_sha256 equals discovery_evidence_index_sha256 and the durability attestation repeats that same digest in both result_payload_sha256 and discovery_evidence_index_sha256_or_null.",
          "full-runtime": "The inner result bytes are exactly the complete canonical temporac.dispatch-result.v5 bytes from the child's sole 0x21 frame including LF. payload_sha256 equals the child success-terminal output_sha256 and durability result_payload_sha256. The separately assembled final trace resolves the same dispatch-result digest through its raw stream and launch/join evidence; the controller neither wraps nor modifies the dispatch result before persistence."
        },
        "noncircularity": "payload_* names the already persisted inner discovery/full result, not this success envelope. The durability attestation binds that inner digest and excludes its own digest and the success envelope; only the later success envelope binds the attestation digest.",
        "output_failure_rule": "Before the first FD1 byte, construct and validate the entire one-frame controller stream in memory. If success-frame construction fails, select OUTPUT and, before writing any byte, construct the exact three-key error stream instead; if an already selected error stream cannot be constructed, emit no bytes and exit124. Write the selected complete stream with an EINTR-retrying loop. Any zero progress, non-EINTR failure or FD1 identity drift before the first byte yields no bytes and exit124; after one or more bytes it closes immediately and exits124 with only that strict prefix and no second frame. After a complete write, close FD1 and require success; close failure exits124 even if the caller observed complete bytes. FD2 receives no byte and is closed. Only a complete selected frame, EOF and its matching 0/122/123 exit is valid.",
        "success_closed_keys": [
          "durability_attestation_sha256",
          "mode",
          "payload_bytes",
          "payload_bytes_hex",
          "payload_sha256",
          "schema"
        ],
        "success_rule": "mode equals the validated request mode. payload_bytes is the positive exact byte count of the complete inner result including LF and is at most 268436480; discovery additionally requires the complete evidence index at most discovery_evidence_index_max_bytes and full requires the complete dispatch-result at most dispatch_result_max_bytes. payload_bytes_hex is lowercase complete inner result bytes of exactly that count; payload_sha256 hashes exactly those bytes and equals the durability attestation result_payload_sha256. durability_attestation_sha256 hashes the exact persisted twelve-key attestation bytes including LF. After canonical serialization the complete six-key success envelope including LF is at most 536874240 and below the one-frame payload cap 1073741824. The envelope has no own digest and is emitted only after durable reopen/rehash; no raw inner output, character count or pre-hex approximation may satisfy a cap.",
        "success_schema": "temporac.native-controller-success.v5"
      },
      "runtime_launcher_artifact_schema": {
        "closed_keys": [
          "child_argv",
          "contract_sha256",
          "controller_argv",
          "cpython_initialization_profile_sha256",
          "cwd",
          "envp",
          "executable_bytes",
          "executable_path",
          "executable_sha256",
          "fd_contract_sha256",
          "gid",
          "launch_instance_schema",
          "module_search_paths",
          "profile_name",
          "schema",
          "transport_contract_sha256",
          "uid",
          "umask_octal"
        ],
        "future_fields": [
          "contract_sha256",
          "cpython_initialization_profile_sha256",
          "executable_bytes",
          "executable_sha256",
          "fd_contract_sha256",
          "transport_contract_sha256"
        ],
        "rule": "This object is the schema and construction authority, not the runtime_launcher value itself. The future-populated 18-key runtime_launcher object governed by this schema is the sole value of cpu-replay-environment.v5 runtime_launcher: its key set is exactly closed_keys; every static key/value is byte-for-byte equal to static_values; every key in future_fields is present and non-null; static_values keys and future_fields are disjoint and their union is exactly closed_keys. contract_sha256 is the accepted Round7 successor. executable_bytes is a positive JSON integer; executable_sha256 hashes the complete installed bytes at executable_path and resolves one native-install native-build-output plus ELF row. cpython_initialization_profile_sha256 and fd_contract_sha256 resolve the exact corresponding BASE roles; transport_contract_sha256 resolves the exact BASE transport artifact whose rows deliberately hash controller subobjects other than this schema descriptor and the later runtime_launcher instance, so there is no cycle. Missing,extra,placeholder,descriptor-as-value,old-v4 command-only object or alternate executable/config value fails.",
        "schema": "temporac.native-isolated-launch-artifact-schema.v5",
        "static_values": {
          "child_argv": [
            "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
            "--embedded-cpython-v4"
          ],
          "controller_argv": [
            "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
            "--controller-v4"
          ],
          "cwd": "/",
          "envp": [
            "BLIS_NUM_THREADS=1",
            "LANG=C",
            "LC_ALL=C",
            "MKL_DYNAMIC=FALSE",
            "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "OMP_DYNAMIC=FALSE",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "PYTHONHASHSEED=0",
            "TZ=UTC",
            "VECLIB_MAXIMUM_THREADS=1"
          ],
          "executable_path": "/opt/temporac/replay-v4/bin/temporac-origin-supervisor",
          "gid": 0,
          "launch_instance_schema": "temporac.native-launch-instance.v5",
          "module_search_paths": [
            "/opt/temporac/replay-v4/lib/python3.12",
            "/opt/temporac/replay-v4/lib/python3.12/lib-dynload",
            "/opt/temporac/replay-v4/lib/python3.12/site-packages"
          ],
          "profile_name": "controller-and-embedded-child-v5",
          "schema": "temporac.native-isolated-launch.v5",
          "uid": 0,
          "umask_octal": "077"
        }
      },
      "side_effect_boundary": "Only the controller writes durable bytes. Dispatch handlers may read exact content-addressed inputs and return bytes; they never persist, rename, fsync, open network, spawn a process, consult clock/random/environment/cwd, or duplicate controller durability."
    },
    "dispatch_abi": {
      "c_call_sequence": [
        "PyImport_ImportModule(\"pams.temporac.protocol_entry\")",
        "PyObject_GetAttrString(module,\"dispatch\")",
        "PyBytes_FromStringAndSize(invocation_bytes,count)",
        "PyObject_CallOneArg(dispatch,bytes_object)",
        "PyBytes_CheckExact(result)",
        "PyBytes_Size and PyBytes_AsString exact copy",
        "strict canonical result parse, digest verification and bytewise re-encoding"
      ],
      "cached_resolution_rule": "The complete pre-dispatch import plan has already imported pams.temporac.protocol_entry and retained its exact sys.modules entry before DISPATCH_BEGIN. The first C-call step is required to return that same exact module object through the cached path and to emit zero raw import/compile/exec audit event and create zero executable mapping; deletion,replacement,reload,loader call or cache miss fails before dispatch. PyObject_GetAttrString must return the exact allowlisted module's dispatch function object, and no descriptor/table digest is read from Python source. Thus the literal C-call sequence does not weaken the post-DISPATCH_BEGIN zero-import-origin rule.",
      "descriptor_construction": {
        "default": {
          "argument_schema": "temporac.invocation.v4",
          "callable": "pams.temporac.protocol_entry:dispatch",
          "continuation_from_or_null": null,
          "external_invocable": true,
          "handler_callable": "pams.temporac.protocol_entry:_dispatch_exact_owner",
          "input_index_schema": "temporac.stage-input-index.v4",
          "max_input_bytes": 1073741824,
          "max_result_bytes": 134217728,
          "result_schema": "temporac.stage-result.v5",
          "side_effect_profile": "read-only-cas-no-network-no-spawn-no-clock-no-random-no-env-no-cwd"
        },
        "handler_token_rule": "handler_token equals owner byte-for-byte; no alias, case fold, prefix, suffix, default or caller-selected callable.",
        "overrides": {
          "G5B-JOIN": {
            "side_effect_profile": "atomic-capability-consume-then-read-only-vault-no-other-side-effect"
          },
          "K7-FINAL": {
            "continuation_from_or_null": "G5B-JOIN",
            "external_invocable": false,
            "side_effect_profile": "same-process-once-after-g5b-read-only-cas-no-network-no-spawn-no-clock-no-random-no-env-no-cwd"
          }
        },
        "row_algorithm": "Iterate owner_order without sorting. For each owner copy every default field, set owner and handler_token to that exact string, then apply only the named override fields. The resulting object must have exactly row_closed_keys; any other override, caller descriptor or dynamic registry fails.",
        "row_closed_keys": [
          "argument_schema",
          "callable",
          "continuation_from_or_null",
          "external_invocable",
          "handler_callable",
          "handler_token",
          "input_index_schema",
          "max_input_bytes",
          "max_result_bytes",
          "owner",
          "result_schema",
          "side_effect_profile"
        ]
      },
      "exception_rule": "Any Python exception, non-exact bytes return, schema failure, wrong owner, oversize result or side-effect violation is cleared with PyErr_Clear and mapped only to the fixed native DISPATCH or RESULT failure stage. No repr, str, traceback, exception type/message or Python stderr is serialized.",
      "function_signature": "def dispatch(invocation_bytes: bytes, /) -> bytes",
      "input_rules": [
        "Exactly one positional argument, no keyword, and type(invocation_bytes) is bytes; bytearray, memoryview, subclass, mapping and path are rejected.",
        "Invocation is exact canonical temporac.invocation.v4 with five keys contract_sha256,input_index_bytes,input_index_sha256,owner,schema. Despite its historical name, input_index_bytes is complete lowercase even-length hexadecimal; decoded bytes hash to input_index_sha256 and parse as the descriptor's exact input_index_schema.",
        "Owner must equal one descriptor row. K7-FINAL cannot be externally invoked and is called exactly once only as the in-process continuation after G5B-JOIN PASS and consumed capability."
      ],
      "owner_order": [
        "P00-IMPLEMENT",
        "P01-STATIC",
        "P02-X0-FIXTURE",
        "P03-TOPO-FIXTURE",
        "P04-OP-FIXTURE",
        "P05-GRAPH-FIXTURE",
        "P05M-COUNT-METRIC",
        "P06-FRESH-PREFLIGHT",
        "S0-COMMIT",
        "G0-ACQUIRE",
        "K0-FIREWALL",
        "temporac.execution.v4/teacher/seed=20260815",
        "temporac.execution.v4/teacher/seed=20260816",
        "temporac.execution.v4/teacher/seed=20260817",
        "G1-TEACHER-AGG",
        "K1-COVERAGE",
        "G2-CONFORMANCE",
        "K2-TARGETS",
        "temporac.execution.v4/response/canonical/seed=20260815",
        "temporac.execution.v4/response/canonical/seed=20260816",
        "temporac.execution.v4/response/canonical/seed=20260817",
        "temporac.execution.v4/response/capacity-control/seed=20260815",
        "temporac.execution.v4/response/capacity-control/seed=20260816",
        "temporac.execution.v4/response/capacity-control/seed=20260817",
        "temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260815",
        "temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260816",
        "temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260817",
        "temporac.execution.v4/response/shortcut/nuisance-only/seed=20260815",
        "temporac.execution.v4/response/shortcut/nuisance-only/seed=20260816",
        "temporac.execution.v4/response/shortcut/nuisance-only/seed=20260817",
        "temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260815",
        "temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260816",
        "temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260817",
        "temporac.execution.v4/response/shortcut/static-code-only/seed=20260815",
        "temporac.execution.v4/response/shortcut/static-code-only/seed=20260816",
        "temporac.execution.v4/response/shortcut/static-code-only/seed=20260817",
        "temporac.execution.v4/response/shortcut/warp-metadata/seed=20260815",
        "temporac.execution.v4/response/shortcut/warp-metadata/seed=20260816",
        "temporac.execution.v4/response/shortcut/warp-metadata/seed=20260817",
        "temporac.execution.v4/response/shortcut/track-length/seed=20260815",
        "temporac.execution.v4/response/shortcut/track-length/seed=20260816",
        "temporac.execution.v4/response/shortcut/track-length/seed=20260817",
        "G3-RESPONSE-AGG",
        "X0I-20260815",
        "X0I-20260816",
        "X0I-20260817",
        "K3-ROUTE",
        "K4-DIAG",
        "G4-OPERATOR",
        "K5-BOUNDARY",
        "K6-RESAMPLER",
        "NATP-20260815",
        "NATP-20260816",
        "NATP-20260817",
        "G5A-COMMIT",
        "G5B-JOIN",
        "K7-FINAL"
      ],
      "result": {
        "inner_output_row_closed_keys": [
          "bytes",
          "bytes_hex",
          "media_type",
          "role",
          "sha256"
        ],
        "inner_rule": "schema is exactly temporac.stage-result.v5 and status is exactly PASS. outputs is nonempty and ordered by strict printable-ASCII role bytes with unique nonempty role strings. Every five-key row has positive bytes, lowercase bytes_hex decoding to exactly that count, one nonempty strict printable-ASCII media_type, and sha256 over those complete bytes. After canonical serialization plus LF the complete stage-result byte string, including all hex expansion and syntax, must be at most the invoked descriptor max_result_bytes=134217728 and framing limits stage_result_max_bytes; the cap is not applied to the sum of decoded outputs alone. The invoked canonical owner procedure validates its already frozen owner-specific artifact/receipt schemas before returning rows; the ABI adds no new scientific output or persistence. Missing, duplicate, non-ASCII, unsorted, count/hash-mismatched, oversize-complete-object or owner-procedure-invalid output fails.",
        "inner_schema": "temporac.stage-result.v5",
        "inner_top_closed_keys": [
          "outputs",
          "schema",
          "status"
        ],
        "wrapper_closed_keys": [
          "owner",
          "result_bytes",
          "result_bytes_hex",
          "result_schema",
          "result_sha256",
          "schema"
        ],
        "wrapper_rule": "owner equals the invoked descriptor owner byte-for-byte; result_schema equals that descriptor's result_schema and is temporac.stage-result.v5; result_bytes is the positive exact byte count of the complete canonical inner object including LF and is at most 134217728; result_bytes_hex is lowercase hex decoding to exactly those bytes; result_sha256 hashes those bytes; schema is temporac.dispatch-result.v5. The complete canonical six-key wrapper including LF, after this one hex expansion, is at most 268436480 and therefore fits one child frame whose payload cap is 1073741824. Parse/re-encode byte equality is mandatory and no field may be caller supplied independently.",
        "wrapper_schema": "temporac.dispatch-result.v5"
      },
      "science_boundary": "A 3.12.13 dispatch handler is a provenance verifier/replayer for the named already frozen owner contract. It may read exact content-addressed historical artifacts, recompute the specifically required bytewise check, and return verification evidence; it never launches training, creates a new scientific artifact, mutates a 3.12.4 receipt, substitutes a recomputation for historical bytes, changes a score/metric, or grants PASS authority. Any recomputed byte difference returns native failure. G5B/K7 ordering and capability semantics remain the existing evaluator verification path, not a new job.",
      "side_effect_profile_definitions": {
        "atomic-capability-consume-then-read-only-vault-no-other-side-effect": "Used only by G5B-JOIN. At the already frozen evaluator-start boundary it atomically consumes the exact one-use grant, then and only then opens and reads the exact committed vault/P402 objects through their validated indexes in the same nonresumable process. It performs no filesystem write, persistence, network, child process, clock, random, environment or cwd access; any failure burns the grant and returns no PASS bytes.",
        "read-only-cas-no-network-no-spawn-no-clock-no-random-no-env-no-cwd": "Used by every descriptor except G5B-JOIN and K7-FINAL. After the supervisor has validated the invocation and exact owner input index, the handler may open only the content-addressed files and typed transitive roles named by that already frozen owner-specific index/contract, must rehash/parse them exactly, and may allocate only process memory for recomputation/result bytes. It performs zero filesystem writes, durable publication, vault access, network, child process, clock, random, environment or cwd read/change; the controller alone persists the returned wrapper.",
        "same-process-once-after-g5b-read-only-cas-no-network-no-spawn-no-clock-no-random-no-env-no-cwd": "Used only by K7-FINAL. It is entered by the internal continuation exactly once after the same process's G5B-JOIN PASS and consumed grant, never by an external invocation; it may read only the exact already validated G5B/vault/CAS bytes permitted by the frozen K7 contract and returns in-memory result bytes. It performs no additional capability consume, filesystem write, persistence, network, child process, clock, random, environment or cwd access."
      },
      "table_artifact": {
        "closed_keys": [
          "contract_sha256",
          "rows",
          "schema"
        ],
        "digest_rule": "dispatch_table_sha256 is SHA-256 of the complete canonical dispatch-table bytes including LF. The digest is external and absent from the table.",
        "row_rule": "rows are exactly the 57 objects produced by descriptor_construction.row_algorithm in owner_order, without sorting, omission, extension or caller input. Every row has exactly descriptor_construction.row_closed_keys and validates all literal/default/override domains.",
        "schema": "temporac.protocol-dispatch-table.v5"
      },
      "table_schema": "temporac.protocol-dispatch-table.v5",
      "universe_rule": "The 57 dispatch descriptors are an ABI universe, not receipt-DAG nodes. Their first 54 owner strings equal the exact pre-G5a roster order and their final three are G5A-COMMIT,G5B-JOIN,K7-FINAL; no receipt edge or 54/106 count is inferred from this table."
    },
    "framing_and_failures": {
      "child_frame": "uint8(type) || uint64_be(sequence) || uint64_be(payload_length) || raw32(SHA256(payload)) || payload; sequence starts at zero and is gap-free. Length and schema are checked before allocation and every payload is the complete canonical JSON object bytes including exactly one LF; inner arbitrary artifact bytes appear only as lowercase hex inside their closed JSON wrapper.",
      "child_stream": {
        "error": "Exactly one type 0x7f frame carrying temporac.native-failure.v5, then EOF. A child failure whose stage is DISPATCH or RESULT exits 121; every other child-stage failure exits 120. Signal/exec/broken-stream cases without an observable child terminal are synthesized only by the controller.",
        "failure_prefix": "Any already completed success-prefix frames remain verifiable, but after the first failure frame there is no other frame or byte. The stream header/failure hash obey exactly three phases: (1) through failure at CHILD_ENTRY_FD_SNAPSHOT,CARRIER_READ,CARRIER_CANONICAL,CARRIER_HASH,CARRIER_CLOSE or LAUNCH_CANONICAL, mode_u8=0 and launch digest=null; (2) after complete canonical launch parsing has selected one exact mode but before successful LAUNCH_HASH, a failure at LAUNCH_HASH uses that exact mode_u8 and digest=null; (3) only after LAUNCH_HASH succeeds, every later child failure uses the exact mode_u8 and exact launch digest. No claimed carrier digest, partial JSON field, default mode or later fallback crosses a phase.",
        "frame_type_table": {
          "0x01": "exactly one temporac.native-launch-carrier-readback.v5 in every success stream",
          "0x02": "exactly one temporac.linux-fd-readback.v5 with stage post-carrier-exec-entry in every success stream",
          "0x03": "exactly one temporac.linux-fd-readback.v5 with stage locked-pre-cpython in every success stream",
          "0x04": "exactly one temporac.linux-fd-readback.v5 with stage post-pyinitialize in every success stream",
          "0x05": "exactly one preserved temporac.cpython-initialization-readback.v4 in every success stream",
          "0x10": "zero or more temporac.raw-audit-event.v5 rows",
          "0x11": "one or more child-origin temporac.normalized-executable-event.v5 rows; PROCESS_EXEC is controller-created and absent",
          "0x12": "exactly two temporac.executable-mapping-set.v5 payloads on success: initial-preinitialize then terminal-after-finalize",
          "0x13": "zero or more temporac.dynamic-executable-mapping-delta.v5 payloads, exactly one after all raw 0x10 rows and before all normalized 0x11 rows of the optional CPython-initialization group and each precommitted nonempty top-level-import group",
          "0x20": "exactly one temporac.generated-source-discovery-output.v5 in discovery and forbidden in full-runtime",
          "0x21": "exactly one temporac.dispatch-result.v5 in full-runtime and forbidden in discovery",
          "0x7e": "exactly one temporac.native-child-success.v5 terminal on success",
          "0x7f": "exactly one temporac.native-failure.v5 terminal on failure and forbidden on success"
        },
        "magic": "ASCII temporac.native-supervisor-stream.v5 || NUL || mode_u8",
        "mode_u8": {
          "0": "unknown-before-carrier-error-only",
          "1": "discovery-run-0",
          "2": "discovery-run-1",
          "3": "full-runtime"
        },
        "success_order": [
          "0x01 carrier readback",
          "one direct 0x11 NATIVE_FD_READBACK for carrier-validated-and-closed, then direct 0x11 PROCESS_IMAGE and every direct 0x11 PREINIT_NATIVE_IMAGE in precommitted order",
          "exactly one 0x12 initial-preinitialize executable mapping set",
          "0x02 post-carrier exec-entry FD readback",
          "one direct 0x11 NATIVE_FD_READBACK for post-carrier-exec-entry",
          "0x03 locked-pre-cpython FD readback",
          "one direct 0x11 NATIVE_FD_READBACK for locked-pre-cpython; then FP lock, audit-buffer allocation and audit-hook installation occur without an output frame",
          "0x04 post-pyinitialize FD readback",
          "one direct 0x11 NATIVE_FD_READBACK for post-pyinitialize",
          "0x05 CPython configuration readback",
          "one direct 0x11 CPYTHON_CONFIG_READBACK",
          "the complete audit_buffer flush of the exact precommitted CPython-initialization prefix: all initialization raw 0x10 rows, its sole 0x13 group when present, then all initialization normalized 0x11 rows in raw-sequence order",
          "each explicit import-plan group in request order: all causal raw 0x10 rows for that one top-level call, then its sole precommitted 0x13 when present, then all normalized 0x11 rows in raw-sequence order; one 0x13 precedes every primary origin that cites its shared digest and no unrelated 0x11 interposes",
          "exactly one 0x20 discovery output for discovery or exactly one 0x21 dispatch result for full-runtime",
          "zero or more 0x10 raw audit-event rows and their mapped 0x11 normalized-event rows caused during Py_FinalizeEx, in causal order; ignored-nonexecutive rows need no normalized row and every reject or post-boundary executable row fails",
          "exactly one 0x12 terminal-after-finalize executable mapping set whose bytes equal sequential delta replay and fresh capture",
          "exactly one 0x7e success terminal",
          "EOF and child exit zero"
        ],
        "universal_fd": 6
      },
      "failure_record": {
        "child_stage_order": [
          "CHILD_ENTRY_FD_SNAPSHOT",
          "CARRIER_READ",
          "CARRIER_CANONICAL",
          "CARRIER_HASH",
          "CARRIER_CLOSE",
          "LAUNCH_CANONICAL",
          "LAUNCH_HASH",
          "BUNDLE_READ",
          "BUNDLE_CANONICAL",
          "BUNDLE_HASH",
          "INITIAL_MAPPING_CAPTURE",
          "FD_EXEC_ENTRY",
          "FD_LOCK",
          "FD_LOCKED_READBACK",
          "FP_LOCK",
          "AUDIT_BUFFER_ALLOC",
          "AUDIT_HOOK",
          "AUDIT_BUFFER_CAPTURE",
          "PY_PREINITIALIZE",
          "PY_CONFIG",
          "PY_INITIALIZE",
          "INITIALIZATION_MAPPING_CAPTURE",
          "POST_INITIALIZE_READBACK",
          "AUDIT_BUFFER_FLUSH",
          "PRE_DISPATCH_IMPORT",
          "IMPORT_GROUP_MAPPING_CAPTURE",
          "TRUSTED_GENERATED_CONSUMPTION",
          "DISPATCH",
          "RESULT",
          "FINALIZE",
          "TERMINAL_MAPPING_CAPTURE",
          "AUDIT_BUFFER_RELEASE",
          "STREAM_WRITE"
        ],
        "closed_keys": [
          "launch_instance_sha256_or_null",
          "schema",
          "stage"
        ],
        "controller_stage_order": [
          "CONTROLLER_REQUEST",
          "DURABLE_REQUEST",
          "ATTEMPT_MARKER",
          "OPERATIONAL_MEMFD_CONSTRUCTION",
          "CARRIER_CONSTRUCTION",
          "PRELAUNCH",
          "FORK",
          "EXEC",
          "STREAM_PARSE",
          "WAITPID",
          "DURABILITY",
          "OUTPUT"
        ],
        "hash_rule": "For a child-emitted failure, launch_instance_sha256_or_null is null for every failure through and including LAUNCH_HASH; it becomes that child's exact digest only after LAUNCH_HASH succeeds. The stream mode byte independently follows child_stream.failure_prefix, including exact mode with null digest only for a LAUNCH_HASH failure after canonical launch parsing. For a controller-synthesized failure attributable to one child, the digest is null before that child's complete controller-side launch plus carrier construction and is that exact digest for PRELAUNCH,FORK,EXEC,STREAM_PARSE or WAITPID even when the child never validated FD7. PRELAUNCH is selected only after both launch and carrier construction have completed and while validating/persisting their closed prelaunch record. A full-runtime aggregate DURABILITY/OUTPUT failure repeats its sole launch digest. A discover-twin aggregate DURABILITY/OUTPUT failure after both joined uses null because two launch digests exist and selecting either is forbidden; request/attempt/operational/carrier-construction failures with no completed current carrier also use null. An earlier discovery run never supplies the hash for a later run's prelaunch failure. No other null/present choice exists.",
        "schema": "temporac.native-failure.v5",
        "stage_assignment_rule": "Each process executes its applicable stage-order array. stage is the first named boundary that fails; a more specific boundary wins over enclosing/later cleanup. Child uses child_stage_order. AUDIT_BUFFER_ALLOC/CAPTURE/FLUSH/RELEASE are disjoint: callback catalog/argument/canonicalization/append failures are CAPTURE and byte drain/write is FLUSH. INITIAL_MAPPING_CAPTURE covers sole initial 0x12. After Py_Initialize returns, expected-initialization raw/normalized sequence,provider,origin and nonmapping equality failures are POST_INITIALIZE_READBACK; only before/after/delta/group/mapping/provider-link/frame failures are INITIALIZATION_MAPPING_CAPTURE. IMPORT_GROUP_MAPPING_CAPTURE covers corresponding mapping boundary for each explicit PyImport; PRE_DISPATCH_IMPORT covers its module/provider/optional-absence/expected-row nonmapping failures. TERMINAL_MAPPING_CAPTURE covers replay,fresh terminal set,sole terminal 0x12. Generated candidate/twin/trusted matching maps only TRUSTED_GENERATED_CONSUMPTION. DISPATCH covers Python call/exception/type and RESULT only returned-byte cap/schema/hash/wrapper. Controller uses controller_stage_order. child_fd_actions/output preservation,reservations,sealed inputs,null-OFD,pipe/FD48,target install map OPERATIONAL_MEMFD_CONSTRUCTION; pre-fork set/record maps PRELAUNCH. Child exit125 maps FORK and controller exit123; child exit126 maps EXEC and controller exit122. Malformed,truncated,short,early-EOF or broken FD6 child bytes map STREAM_PARSE and exit122. Signal/nonprotocol child exit or child peer-close proof maps WAITPID and exit122; waitpid syscall, child-identity or join-evidence mismatch maps WAITPID and exit123. Persistence outside request/attempt/prelaunch maps DURABILITY. Result/stderr restore or serialization/write/close maps OUTPUT if no earlier token; FD1/FD32 failure exits124, while isolated FD33 restore failure emits through valid FD1 and exits123, retaining any already selected earlier stage/payload rather than replacing it with OUTPUT. Cleanup never replaces selected token; no errno/message/alternate/second failure changes it.",
        "stage_tokens": [
          "CHILD_ENTRY_FD_SNAPSHOT",
          "CARRIER_READ",
          "CARRIER_CANONICAL",
          "CARRIER_HASH",
          "CARRIER_CLOSE",
          "LAUNCH_CANONICAL",
          "LAUNCH_HASH",
          "BUNDLE_READ",
          "BUNDLE_CANONICAL",
          "BUNDLE_HASH",
          "INITIAL_MAPPING_CAPTURE",
          "FD_EXEC_ENTRY",
          "FD_LOCK",
          "FD_LOCKED_READBACK",
          "FP_LOCK",
          "AUDIT_BUFFER_ALLOC",
          "AUDIT_HOOK",
          "AUDIT_BUFFER_CAPTURE",
          "PY_PREINITIALIZE",
          "PY_CONFIG",
          "PY_INITIALIZE",
          "INITIALIZATION_MAPPING_CAPTURE",
          "POST_INITIALIZE_READBACK",
          "AUDIT_BUFFER_FLUSH",
          "PRE_DISPATCH_IMPORT",
          "IMPORT_GROUP_MAPPING_CAPTURE",
          "TRUSTED_GENERATED_CONSUMPTION",
          "DISPATCH",
          "RESULT",
          "FINALIZE",
          "TERMINAL_MAPPING_CAPTURE",
          "AUDIT_BUFFER_RELEASE",
          "STREAM_WRITE",
          "CONTROLLER_REQUEST",
          "DURABLE_REQUEST",
          "ATTEMPT_MARKER",
          "OPERATIONAL_MEMFD_CONSTRUCTION",
          "CARRIER_CONSTRUCTION",
          "PRELAUNCH",
          "FORK",
          "EXEC",
          "STREAM_PARSE",
          "WAITPID",
          "DURABILITY",
          "OUTPUT"
        ]
      },
      "limits": {
        "caps_rule": "Every cap applies to the complete serialized bytes named, including JSON syntax and terminal LF, before hex embedding. A stage-result is at most 134217728 bytes; its dispatch-result wrapper is independently at most 268436480; either discovery-evidence or dispatch-result used as a controller inner result is at most 268436480; the complete controller-success wrapper is independently at most 536874240. Thus each single hex embedding plus fixed fields must pass the next explicit cap and both child/controller frame payloads remain below 1073741824. No cap is interpreted as raw output bytes, character count, pre-hex bytes at the wrong layer or allocation permission.",
        "child_stream_max_bytes": 4294967295,
        "controller_inner_result_max_bytes": 268436480,
        "controller_request_max_bytes": 68719476735,
        "controller_success_max_bytes": 536874240,
        "discovery_evidence_index_max_bytes": 268436480,
        "dispatch_input_max_bytes": 1073741824,
        "dispatch_result_max_bytes": 268436480,
        "frame_payload_max_bytes": 1073741824,
        "ledger_bundle_max_bytes": 17179869184,
        "single_ledger_row_max_bytes": 16777216,
        "stage_result_max_bytes": 134217728
      },
      "os_failure_rule": "If exec fails, a signal kills the child, FD6 is malformed,truncated,short,early-EOF or broken, or the child cannot write its error FD, the controller synthesizes the same three-key failure object using the known launch hash when available and, absent the exact isolated FD33-restoration exception, exits122 after its one complete error frame and EOF. A fork/child-pre-exec close failure is controller-origin and exits123. If controller FD1 or FD32 is unusable no success/error bytes can be promised; caller observes EOF and controller exit124. If only FD33 or its restoration is unusable while FD1/FD32 remain valid, restoration and result_stream rules require one exact error frame and exit123 while retaining an earlier stage/payload if one exists. Child stdout and stderr are /dev/null; controller stderr is /dev/null and controller stdout carries only the result stream after durability.",
      "success_terminal": {
        "closed_keys": [
          "launch_instance_sha256",
          "mode",
          "output_sha256",
          "schema",
          "status",
          "terminal_executable_mapping_set_sha256"
        ],
        "rule": "status=PASS. mode and launch hash equal the validated launch. output_sha256 hashes the unique earlier discovery-output or dispatch-result payload bytes, not any intervening finalization audit row. terminal_executable_mapping_set_sha256 hashes the immediately preceding exact 0x12 terminal set and equals sequential mapping-delta replay plus fresh capture. The child emits this frame only after successful Py_FinalizeEx, after every raw/mapped finalization event and terminal mapping set have been framed and validated, after audit-buffer release, and after all audit/trusted/dispatch checks; it then closes FD6 and exits zero without another byte.",
        "schema": "temporac.native-child-success.v5"
      },
      "transport_io": "All reads/writes are complete EINTR-safe loops; short EOF is failure; trailing bytes are forbidden; no concatenated JSON, alternate endian, implicit length, stdio buffering or shared discovery stream exists."
    },
    "generated_source_closure": {
      "capsule": {
        "closed_keys": [
          "base_bundle_sha256",
          "contract_sha256",
          "import_plan_sha256",
          "mode",
          "schema",
          "supervisor_sha256"
        ],
        "rule": "mode is exactly pre-dispatch-twin-discovery; every digest resolves through the already committed BASE bundle and installed native ledger. The capsule has no output, precommit, environment, FINAL, launch, evidence, trusted-row, receipt, data or artifact digest and is canonical JSON plus LF.",
        "schema": "temporac.generated-source-discovery-capsule.v5"
      },
      "controller_input": {
        "closed_keys": [
          "capsule_bytes",
          "capsule_bytes_hex",
          "capsule_sha256",
          "precommit_bytes",
          "precommit_bytes_hex",
          "precommit_sha256",
          "schema"
        ],
        "rule": "This single canonical wrapper, and no concatenation or pair of objects, is the discovery controller request payload. Both byte counts are positive; each lowercase hex field decodes to exactly that many complete canonical bytes including LF and each SHA-256 hashes those bytes. The capsule and precommit parse/re-encode byte-identically; the precommit embeds the same complete capsule bytes/count/hash and created_before_process_count=0.",
        "schema": "temporac.discovery-controller-input.v5"
      },
      "discovery": [
        "The exact import plan, static source allowlist, installed distributions, CPython runtime, native images, audit catalog, dispatch table and transport contract are committed in BASE before either discovery process. Discovery cannot teach or backfill BASE.",
        "Twin discovery begins after Py_InitializeFromConfig and all native/config/FD readbacks but before any pams import. Each fresh process executes the same exact pre-dispatch import plan and no dispatch, data, job, artifact, vault or experiment access.",
        "Each output is complete and nonempty. Run0 and run1 canonical outputs, row arrays, hashes and tagged roots must be byte-identical; process/pipe/launch evidence must be distinct where instance-specific. Any nondeterminism fails P06.",
        "Only after twin equality is the trusted index created from exactly the common ordered rows. It binds BASE, never FINAL. Full runtime with FINAL reruns the same import plan and consumes every trusted row once before DISPATCH_BEGIN; after DISPATCH_BEGIN generated compile/exec multiplicity is zero."
      ],
      "evidence_index": {
        "attestation_closed_keys": [
          "base_bundle_sha256",
          "capsule_sha256",
          "carrier_construction_sha256",
          "carrier_readback_sha256",
          "child_pid",
          "child_starttime_ticks",
          "configuration_readback_sha256",
          "controller_request_sha256",
          "discovery_output_root_sha256",
          "discovery_output_sha256",
          "exec_entry_fd_readback_sha256",
          "exit_code",
          "exit_kind",
          "launch_instance_sha256",
          "locked_pre_cpython_fd_readback_sha256",
          "post_pyinitialize_fd_readback_sha256",
          "precommit_sha256",
          "process_ordinal",
          "run_id",
          "schema",
          "signal_or_null",
          "status",
          "stream_eof_observed",
          "stream_sha256",
          "supervisor_sha256",
          "wait_status_u32"
        ],
        "attestation_rule": "The attestation is constructed by the controller only after exact terminal frame, EOF and waitpid of the named child. exit_kind=exited, exit_code=0, signal_or_null=null, wait_status_u32=0, stream_eof_observed=true and status=JOINED_PASS. It contains no own digest. attestation_sha256 hashes its standalone canonical bytes including LF.",
        "attestation_schema": "temporac.generated-source-discovery-run-attestation.v5",
        "digest_rule": "discovery_evidence_index_sha256 is SHA-256 of the complete canonical index bytes including LF, whose complete byte count including every embedded hex field is at most 268436480. discovery_evidence_root_sha256 is SHA-256 of ASCII temporac.generated-source-discovery-evidence-root.v5, one NUL, uint64_be(index byte count), then the exact index bytes. Neither digest occurs inside the index. The complete index, not decoded output bytes alone, is the controller inner payload and must fit both discovery_evidence_index_max_bytes and controller_inner_result_max_bytes before any success envelope is allocated.",
        "external_resolvers": {
          "child_stream_index": "child_stream_index_sha256 resolves only /var/lib/temporac/replay-v4/native-controller-v4/streams/<child_stream_index_sha256>.bin. Reopen O_RDONLY|O_CLOEXEC|O_NOFOLLOW, require regular owner0:0 mode0600, complete bytes/EOF/hash, strict temporac.native-child-stream-index.v5 parse and byte-identical reserialization. mode=discover-twin and rows are exactly ordinals 0,1; each evidence row's stream_bytes/stream_sha256/launch/process_ordinal and joined attestation equal that exact index row, whose bytes_hex supplies the complete magic-through-terminal stream and eof_observed=true. The evidence index never embeds a second copy of those potentially large bytes.",
          "controller_request": "controller_request_sha256 resolves only /var/lib/temporac/replay-v4/native-controller-v4/requests/<controller_request_sha256>.json under the same durable root checks. Reopen and require owner0:0 mode0600, complete bytes/EOF/hash, strict eleven-key temporac.native-controller-request.v5 parse and byte-identical reserialization; mode=discover-twin, bundle is the same BASE digest, environment triple is null and payload is the exact controller-input wrapper binding the same capsule/precommit. Both evidence rows and attestations repeat this one digest. The evidence index never embeds request bytes.",
          "retention": "Both immutable resolver targets and the nonretry attempt marker are retained through FINAL construction and K7; the discover-twin durability attestation binds controller_request_sha256 and child_stream_index_sha256. Missing, mutable, alternative-root/path, digest-only-without-bytes, wrong mode/row, or garbage-collected target fails."
        },
        "row_bindings": [
          "run_id and process_ordinal",
          "BASE bundle bytes/hash and child readback",
          "the one shared durable discover-twin controller request resolver/hash",
          "launch instance bytes/hash",
          "FD7 carrier eight-stage construction bytes/hash and child carrier readback bytes/hash",
          "three operational FD readback bytes/hashes",
          "CPython config readback bytes/hash",
          "the complete durable child-stream-index resolver plus exact per-run byte count/hash/EOF projection",
          "discovery output bytes/hash/root",
          "process identity, terminal frame, wait status and joined attestation"
        ],
        "row_closed_keys": [
          "attestation_bytes_hex",
          "attestation_sha256",
          "base_bundle_sha256",
          "capsule_sha256",
          "carrier_construction_bytes_hex",
          "carrier_construction_sha256",
          "carrier_readback_bytes_hex",
          "carrier_readback_sha256",
          "configuration_readback_bytes_hex",
          "configuration_readback_sha256",
          "controller_request_sha256",
          "discovery_output_bytes_hex",
          "discovery_output_root_sha256",
          "discovery_output_sha256",
          "exec_entry_fd_readback_bytes_hex",
          "exec_entry_fd_readback_sha256",
          "launch_evidence_index_sha256",
          "launch_instance_bytes_hex",
          "launch_instance_sha256",
          "locked_pre_cpython_fd_readback_bytes_hex",
          "locked_pre_cpython_fd_readback_sha256",
          "post_pyinitialize_fd_readback_bytes_hex",
          "post_pyinitialize_fd_readback_sha256",
          "precommit_sha256",
          "process_ordinal",
          "run_id",
          "schema",
          "stream_bytes",
          "stream_sha256",
          "supervisor_sha256"
        ],
        "row_order": [
          0,
          1
        ],
        "row_rule": "Every bytes_hex field retained in the row is lowercase complete bytes and matches its paired hash; every canonical JSON payload includes exactly one LF. stream_bytes is a positive count and stream_sha256 resolves the exact child-stream-index row through external_resolvers rather than an inline hex duplicate. Carrier construction is external to launch and its final projection must equal child carrier readback after allowing only the explicitly stage-specific offset/CLOEXEC transitions. The three operational FD readbacks are individually exact and bind the same launch. Each completed in-memory run attestation binds exactly every per-run digest and instance identity listed by attestation_closed_keys without its own digest. The combined launch_evidence_index_sha256 is intentionally absent from each completed in-memory run attestation because that index is constructed only after all applicable rows; it is bound identically by the evidence-index top level and each evidence row, and its own rows bind the same launch/carrier/operational evidence without an attestation backedge.",
        "row_schema": "temporac.generated-source-discovery-evidence-row.v5",
        "schema": "temporac.generated-source-discovery-evidence-index.v5",
        "top_level_closed_keys": [
          "base_bundle_sha256",
          "capsule_sha256",
          "child_stream_index_sha256",
          "contract_sha256",
          "controller_request_sha256",
          "launch_evidence_index_sha256",
          "precommit_sha256",
          "rows",
          "schema",
          "shared_output_root_sha256",
          "shared_output_sha256"
        ],
        "twin_equality": "Exactly two rows ordered 0,1 with run_id=process_ordinal. controller_request_sha256 resolves one exact durable eleven-key discover-twin request and child_stream_index_sha256 resolves one exact durable two-row stream index under external_resolvers; both evidence rows and attestations repeat the one request digest and their stream count/hash projections equal the corresponding stream-index rows. BASE/capsule/precommit/supervisor and complete output bytes/hash/root are identical. The single launch-evidence index has exactly those two process rows. Launch mode/hash, carrier construction, process identity, pipe, stream and join evidence are distinct. After replacing only launch_instance_sha256 by 64 zeroes in stage readbacks, the three stage projections are byte-identical; no other semantic difference is allowed. Missing, extra, reordered, reused process/OFD/pipe, unjoined, unretained resolver bytes, output-only or digest-only evidence fails."
      },
      "fail_closed_runtime_condition": "A generated compile without a source-backed Python frame, a native-extension generator with no typed observable origin, a delayed or unmatched compile-to-exec pair, different twin output, or a generated action after the trusted prefix is not authorized by inference; P06 fails and requires a separately reviewed predefined-method refactor.",
      "import_plan": [
        "dataclasses",
        "numpy",
        "scipy",
        "torch",
        "pams",
        "pams.types",
        "pams.temporac",
        "pams.temporac._fp_control",
        "pams.temporac.certify",
        "pams.temporac.contract",
        "pams.temporac.cue",
        "pams.temporac.decode",
        "pams.temporac.evaluator",
        "pams.temporac.feature_io",
        "pams.temporac.fixtures",
        "pams.temporac.gates",
        "pams.temporac.hashio",
        "pams.temporac.metrics",
        "pams.temporac.nola",
        "pams.temporac.objective",
        "pams.temporac.prediction",
        "pams.temporac.preprocess",
        "pams.temporac.protocol_entry",
        "pams.temporac.quadrature",
        "pams.temporac.receipts",
        "pams.temporac.response",
        "pams.temporac.runtime",
        "pams.temporac.teacher",
        "pams.temporac.training",
        "pams.temporac.trusted_packer",
        "pams.temporac.types",
        "pams.temporac.x0"
      ],
      "import_plan_artifact": {
        "closed_keys": [
          "contract_sha256",
          "expected_dynamic_load_groups",
          "expected_import_occurrences",
          "expected_initialization_normalized_rows",
          "expected_initialization_raw_rows",
          "installed_distribution_member_index_sha256",
          "meta_path_provider_descriptors",
          "modules",
          "optional_absence_rows",
          "path_hooks_provider_descriptors",
          "schema",
          "source_allowlist_sha256"
        ],
        "digest_rule": "import_plan_sha256 is SHA-256 of the complete canonical artifact bytes including LF. The digest is external and absent from the artifact.",
        "expected_dynamic_load_group_closed_keys": [
          "dynamic_mapping_delta_sha256",
          "group_ordinal",
          "owner_kind",
          "primaries",
          "request_module_or_null",
          "request_ordinal_or_null"
        ],
        "expected_dynamic_load_group_primary_closed_keys": [
          "elf_sha256",
          "normalized_event",
          "origin_sha256",
          "raw_event_sequence"
        ],
        "expected_dynamic_load_group_rule": "expected_dynamic_load_groups is a complete future-populated array ordered by gap-free group_ordinal. owner_kind is cpython-initialization or top-level-import. At most one first cpython-initialization group exists; it has both request fields null and spans the already emitted initial-preinitialize set through successful Py_InitializeFromConfig return. Every other group is top-level-import, corresponds to exactly one modules[request_ordinal_or_null] call, and has request_module_or_null byte-equal that module. Only a boundary containing at least one successful primary executable load has a row. primaries is nonempty and strict raw_event_sequence order; normalized_event is PY_EXTENSION_LOAD or CTYPES_DLOPEN, origin_sha256 hashes the exact complete primary v5 origin including the group's same dynamic_mapping_delta_sha256, and elf_sha256 equals that origin's elf_sha256 field. Initialization primary rows match expected_initialization projections; later extension primaries match expected_import_occurrences and later ctypes rows derive from precommitted installed control flow. Every raw sequence is unique across groups. dynamic_mapping_delta_sha256 is independently derived before BASE from the exact CPython/install ELF/DT_NEEDED closure, expected raw sequences and deterministic mapping-set projections; the delta contains no origin/import-plan digest, so no cycle exists. Runtime constructs the initialization delta at the post-initialize barrier or one later delta at return from the named PyImport call and requires exact bytes/digest/origins. Extra, missing, cross-boundary, reordered, post-dispatch or self-digest-bearing group input fails.",
        "expected_import_occurrence_row_closed_keys": [
          "dynamic_mapping_load_group_ordinal_or_null",
          "dynamic_mapping_primary_raw_sequence_or_null",
          "dynamic_mapping_role_or_null",
          "form",
          "module_name",
          "normalized_event_or_null",
          "occurrence_ordinal",
          "provider_descriptor_ordinal_or_null",
          "raw_event_sequence",
          "reason_token_or_null",
          "resolved_origin_kind_or_null",
          "resolved_origin_sha256_or_null"
        ],
        "expected_import_occurrence_rule": "expected_import_occurrences is a complete future-populated nonempty array ordered by strictly increasing raw_event_sequence; every row has exactly expected_import_occurrence_row_closed_keys. raw_event_sequence is the exact absolute same-process raw audit counter value and occurrence_ordinal is the zero-based occurrence for module_name. form is search or resolved-provider. normalized_event_or_null is one of PY_BUILTIN_IMPORT,PY_FROZEN_IMPORT,PY_IMPORT_SOURCE,PY_EXTENSION_LOAD or null; null requires one exact optional-absence reason and every provider/origin/dynamic field null. A normalized row requires reason null, exact resolved_origin_kind and nonnull resolved_origin_sha256_or_null hashing the complete exact v5 origin selected from the precommitted runtime/install closure, including the exact precommitted group delta digest for extension origins; kind-only or module-only matching is forbidden. provider_descriptor_ordinal_or_null selects the exact meta-path descriptor for search success and is null for resolved-provider. Extension rows alone use role primary/mirror, exact primary raw sequence and nonnull group ordinal matching exactly one group; all other dynamic fields are null. Rows are derived from exact installed source/CPython/import-plan control flow and independently reviewed before BASE, never learned from twin discovery. Every discovery and full child must reproduce this complete import-row projection byte-for-byte after removing only process_ordinal; extra, missing, reordered, changed exact origin/outcome/provider/form/role/group or different raw counter fails.",
        "expected_initialization_normalized_row_closed_keys": [
          "event",
          "origin",
          "origin_sha256",
          "raw_event_sequence_or_null",
          "sequence"
        ],
        "expected_initialization_projection_rule": "expected_initialization_raw_rows and expected_initialization_normalized_rows are complete future-populated arrays for every PySys audit callback and its permitted normalized projection from successful hook installation through Py_InitializeFromConfig return, before the supervisor's first explicit import-plan call. A raw projection has exact keys args,args_sha256,event,schema,sequence and equals temporac.raw-audit-event.v5 after removing only process_ordinal. A normalized projection has exactly expected_initialization_normalized_row_closed_keys and equals a child normalized row after removing process_ordinal and sha256; verifier reinserts the actual process ordinal and recomputes that row hash. Arrays retain causal/raw order, raw sequence is absolute zero-based/gap-free from hook install, normalized sequence equals the fixed native-prefix continuation, and ignored-nonexecutive raws remain only in the raw array. Every import/compile/exec/ctypes callback and complete origin is statically derived from the exact CPython 3.12.13 initialization/config/runtime closure and independently reviewed before BASE. Initialization extension/ctypes primaries match the optional first dynamic-load group. Twin/full children reproduce both arrays byte-for-byte; an extra/missing/reordered event, executable mapping outside that group or generated source during initialization fails.",
        "expected_initialization_raw_row_closed_keys": [
          "args",
          "args_sha256",
          "event",
          "schema",
          "sequence"
        ],
        "optional_absence_row_closed_keys": [
          "module",
          "occurrence_ordinal",
          "reason_token"
        ],
        "optional_absence_rule": "Rows are ordered by ASCII module then nonnegative occurrence_ordinal and are unique. occurrence_ordinal is the zero-based count among raw import events for that exact module in the fixed pre-dispatch execution. Rows are statically proved from exact installed source control flow before BASE. reason_token is exactly platform-inapplicable or explicitly-optional-uninstalled. A row authorizes only that one module occurrence's failed import at its safe barrier; it authorizes no executable origin and may not be learned from discovery. Any unlisted/reordered/reused failed import, present provider for an absence row, or branch whose outcome depends on data/clock/random/environment fails.",
        "rule": "contract_sha256 is accepted Round7 successor; modules equals generated_source_closure.import_plan in order; source/installed digests resolve precommitted closure. Provider descriptor arrays are complete ordered exact rows from audit argument projection, future-populated from accepted installed/runtime closure and reviewed before BASE; every search raw import reproduces them. expected_initialization arrays close the pre-PyInitialize-return prefix; expected_import_occurrences close later import order/outcome; expected_dynamic_load_groups close every initialization or nested pre-dispatch extension/ctypes primary and its unique boundary. optional_absence_rows is complete reviewed finite array fixed before BASE. Artifact has no BASE/FINAL hash,discovery output or runtime-learned value.",
        "schema": "temporac.pre-dispatch-import-plan.v5"
      },
      "import_plan_rule": "Order is exact and module names are duplicate-free. The bound pams package import eagerly imports pams.types; the following explicit pams.types request must therefore be a verified cache hit with no second source compile/exec and is retained to make requested-module order explicit. The bound pams.temporac package root executes its exact docstring, future-annotations import and typed empty __all__, but performs no eager implementation-submodule import; it is no-eager-import, not an empty file. The explicit pams.temporac._fp_control row then loads the native-build-extension from its exact native-install/ELF row before the remaining pams.temporac implementation modules and yields its required PY_EXTENSION_LOAD origin; it is not a source-allowlist row or wheel member. Immediately before each listed top-level request the supervisor captures that group's before mapping set; immediately after the same PyImport call returns it captures after, consumes the exact expected import/ctypes rows and emits at most one precommitted grouped 0x13 before any normalized rows from that group. Every transitive import and every cached/new distinction must resolve through the precommitted installed/source ledger and exact raw-event/group sequence. Optional absence is allowed only by an exact BASE optional-absence row; no catch-all optional import, filesystem enumeration or discovery-learned module exists. Before DISPATCH_BEGIN all expected groups and trusted-generated rows are exhausted; thereafter extension import, ctypes.dlopen, compile of generated source or any executable mapping change is forbidden, so handler execution requires no unresolved loader boundary.",
      "output": {
        "closed_keys": [
          "base_bundle_sha256",
          "capsule_sha256",
          "rows",
          "schema"
        ],
        "hash_rule": "output_sha256 is SHA-256 of the exact canonical output bytes including LF. output_root_sha256 is SHA-256 of ASCII temporac.generated-source-discovery-output-root.v5, one NUL, raw32(capsule_sha256), uint64_be(output byte count), then the exact output bytes. Both fresh outputs must be byte-identical and nonempty; neither digest is embedded in the output.",
        "schema": "temporac.generated-source-discovery-output.v5"
      },
      "precommit": {
        "closed_keys": [
          "capsule_bytes",
          "capsule_bytes_hex",
          "capsule_sha256",
          "contract_sha256",
          "created_before_process_count",
          "schema"
        ],
        "rule": "created_before_process_count is integer zero. capsule_bytes is positive, capsule_bytes_hex is lowercase complete capsule bytes, and capsule_sha256 hashes them. These exact canonical precommit bytes are durably committed only as the validated nested bytes in the complete controller input and rehashed requests/<controller_request_sha256>.json object before the controller may create either discovery child; after reopening that request the controller reextracts and bytewise revalidates the input, precommit and capsule. No separate precommit pathname or mutable side object exists. They contain no output, launch, evidence, trusted, FINAL or environment value.",
        "schema": "temporac.generated-source-discovery-precommit.v5"
      },
      "row": {
        "closed_keys": [
          "compile_filename",
          "generated_source_bytes",
          "generated_source_hex",
          "generated_source_sha256",
          "generator_origin",
          "generator_origin_sha256",
          "kind",
          "ordinal",
          "profile",
          "schema",
          "source_type"
        ],
        "origin_variants": [
          "cpython-stdlib",
          "project-installed",
          "installed-wheel"
        ],
        "rules": [
          "generated_source_bytes is positive, generated_source_hex decodes to exactly that many bytes and generated_source_sha256 hashes those bytes.",
          "The immediately matched raw compile projection is exact temporac.audit-args.compile.v5: compile_filename equals its filename, generated_source_bytes/source_hex/source_sha256/source_type equal its source_bytes/source_hex/source_sha256/source_type byte-for-byte, str-utf8 means one strict UTF-8 encoding with no BOM and no surrogate, and bytes means type(source) is exactly bytes and the buffer is used verbatim.",
          "generator_origin is a complete typed canonical origin object; generator_origin_sha256 hashes that standalone object including LF and is never equated to a member digest. The origin embeds the resolved installed/runtime member digest.",
          "At the raw compile callback call PyEval_GetFrame exactly once, then inspect that current innermost frame followed by PyFrame_GetBack in order, at most 4096 frames. For each visited frame, PyFrame_GetCode and its exact code-authority/co_filename must resolve through the precommitted trace rules; cpython-core builtin/frozen frames are nonselectable and traversed, while the first frame resolving exactly one cpython-stdlib, project-installed or installed-wheel source origin is selected. An unresolved, trusted-generated, native-extension or otherwise ineligible executable frame before selection, no selectable frame, depth overflow, cycle/repeated frame identity, API error or ambiguous ledger match fails. The selected complete v5 origin is generator_origin; qualname, repr, address, outer-frame preference and caller-supplied origin never participate.",
          "kind is exactly non-file-backed-python-source, profile is pre-dispatch-import-plan-v5, source_type is str-utf8 or bytes, and schema is temporac.trusted-generated-source-row.v5.",
          "Rows are ordinal 0..R-1 and each compile is immediately followed by its matching exec in the projection containing only generated compile/exec actions. CPython exposes no compile mode, so compile_mode is absent."
        ],
        "schema": "temporac.trusted-generated-source-row.v5"
      },
      "scope_evidence": "The final bound current tree contains 89 @dataclass decorator sites (87 under pams.temporac and 2 under pams.types), proving the Round6 dataclasses-only three-file assumption insufficient. Eighty-nine is a source-site observation, not an asserted generated-row count.",
      "trusted_index": {
        "row_closed_keys": [
          "compile_filename",
          "generated_source_bytes",
          "generated_source_hex",
          "generated_source_sha256",
          "generator_origin",
          "generator_origin_sha256",
          "kind",
          "ordinal",
          "profile",
          "schema",
          "source_type",
          "trusted_row_sha256"
        ],
        "row_hash_rule": "trusted_row_sha256 is SHA-256 of the exact standalone eleven-key generated-source row in generated_source_closure.row.closed_keys, canonical JSON plus LF. The digest is then appended as the sole twelfth trusted-index row key. It is unique and may not substitute for any direct row field.",
        "row_rule": "Rows are exactly the complete nonempty common discovery-output rows in ordinal order with only the independently recomputed trusted_row_sha256 added. Full runtime reconstructs and consumes each exact eleven-key preimage once in the adjacent compile/exec pair; no row is learned, filtered, reordered, duplicated or left after DISPATCH_BEGIN.",
        "schema": "temporac.trusted-generated-source-index.v5",
        "top_level_closed_keys": [
          "base_bundle_sha256",
          "discovery_evidence_index_sha256",
          "discovery_evidence_root_sha256",
          "discovery_output_root_sha256",
          "rows",
          "schema"
        ]
      }
    },
    "ledger_bundles": {
      "base_role_order": [
        "audit-event-policy-catalog",
        "cpython-initialization-profile",
        "cpython-runtime-member-index",
        "dispatch-table",
        "effective-contract-index",
        "elf-index",
        "fd-contract",
        "floating-point-control-profile",
        "initial-native-image-index",
        "installed-distribution-member-index",
        "kernel-runtime-evidence",
        "native-build-recipe",
        "native-rebuild-evidence",
        "native-install-index",
        "native-transport-contract",
        "pre-dispatch-import-plan",
        "wheel-index",
        "source-allowlist",
        "version-layer-index"
      ],
      "bundle": {
        "closed_keys": [
          "contract_sha256",
          "rows",
          "schema"
        ],
        "row_closed_keys": [
          "bytes",
          "bytes_hex",
          "role",
          "sha256"
        ],
        "row_rule": "bytes is a positive integer; bytes_hex is lowercase complete canonical artifact bytes of exactly that count; sha256 hashes those bytes; role is unique and appears in the exact role order. Every JSON role is duplicate-free canonical UTF-8 with one LF and re-encodes byte-identically.",
        "schema": "temporac.runtime-ledger-bundle.v5"
      },
      "carrier": "FD8 is the sole bundle carrier in every child. Discovery carries exact BASE bytes; full carries exact FINAL bytes. The controller request contains the same complete bytes/count/hash, the launch binds the hash, and child pread/readback must equal all three before Python initialization.",
      "final_environment": {
        "closed_keys": [
          "architecture",
          "autocast",
          "blas",
          "code_index_sha256",
          "contract_sha256",
          "cpu_dispatch",
          "cpu_isa",
          "cpython_initialization_profile_sha256",
          "cpython_runtime_member_index_sha256",
          "determinism",
          "elf_index_sha256",
          "environment_variables",
          "fd_contract_sha256",
          "floating_point_control",
          "generated_source_discovery_capsule_sha256",
          "generated_source_discovery_evidence_index_sha256",
          "generated_source_discovery_evidence_root_sha256",
          "generated_source_discovery_output_root_sha256",
          "generated_source_discovery_precommit_sha256",
          "installed_distribution_member_index_sha256",
          "kernel_runtime_evidence_sha256",
          "native_rebuild_evidence_index_sha256",
          "numpy",
          "onednn",
          "platform_system",
          "python",
          "runtime_launcher",
          "runtime_ledger_bundle_sha256",
          "schema",
          "scipy",
          "threads",
          "torch",
          "trusted_generated_source_index_sha256",
          "wheel_index_sha256"
        ],
        "preservation": "architecture,autocast,blas,cpu_dispatch,cpu_isa,determinism,environment_variables,numpy,onednn,platform_system,python,scipy,threads,torch retain the exact accepted Round6 value trees and 10+64 semantics. floating_point_control retains the exact accepted Round6 value tree and FP semantics except for the single helper/origin literal replaced by fp_native_abi.floating_point_profile_overlay; the complete effective profile is the exact BASE floating-point-control-profile bytes. code_index_sha256 is exactly the source-allowlist BASE-row sha256, preserving the established code-index meaning rather than referring to a post-run trace/completion index. kernel_runtime_evidence_sha256 resolves the exact BASE ten-key kernel/runtime artifact that authorizes only fixed [vdso]/[vsyscall] mapping rows; native_rebuild_evidence_index_sha256 resolves the exact BASE twin-build evidence and typed output rows used by native-install. Every other digest field resolves the correspondingly named exact v5 BASE/FINAL or generated artifact; runtime_launcher is the exact closed controller/child v5 configuration, not the Round6 command string.",
        "rule": "Construct only after FINAL bytes exist. runtime_ledger_bundle_sha256 is exact SHA-256(FINAL bytes); FINAL excludes every environment byte/hash. environment_sha256 is external SHA-256 of the complete canonical v5 object including LF. The full controller request carries these exact environment bytes/count/hash separately from FINAL, and launch plus child readback bind both without backfill.",
        "schema": "temporac.cpu-replay-environment.v5"
      },
      "final_role_order": [
        "audit-event-policy-catalog",
        "cpython-initialization-profile",
        "cpython-runtime-member-index",
        "dispatch-table",
        "effective-contract-index",
        "elf-index",
        "fd-contract",
        "floating-point-control-profile",
        "initial-native-image-index",
        "installed-distribution-member-index",
        "kernel-runtime-evidence",
        "native-build-recipe",
        "native-rebuild-evidence",
        "native-install-index",
        "native-transport-contract",
        "pre-dispatch-import-plan",
        "wheel-index",
        "source-allowlist",
        "version-layer-index",
        "generated-source-discovery-capsule",
        "generated-source-discovery-evidence-index",
        "generated-source-discovery-output",
        "generated-source-discovery-precommit",
        "trusted-generated-source-index"
      ],
      "noncircularity": [
        "FINAL begins with the exact byte-for-byte 19-row BASE prefix.",
        "Neither bundle contains final CPU environment bytes or hash. The full controller request carries final environment separately; that environment binds FINAL.",
        "Trusted-generated rows bind BASE only; no trusted row binds FINAL, environment, launch, trace, capability or future receipt.",
        "The dispatch-table artifact and protocol_entry source are independent siblings in BASE: both are derived from this accepted specification, source bytes contain no dispatch-table or BASE digest, and the table binds only callable/token/schema values rather than the source digest. Runtime verifies behavior and installed-source equality without either artifact containing the other's hash.",
        "A bundle never contains its own bytes/hash. Its external SHA-256 is computed only after canonical serialization."
      ],
      "topological_order": [
        "accepted effective contract",
        "source bytes, project-build-input bytes, pure immutable toolchain/build-image ledger, kernel-runtime evidence and normative-spec inputs",
        "independently reviewed static native-build recipe binding those prior immutable inputs",
        "two non-authoritative CPython preflights, complete twin equality and retained selected run0 preflight stage",
        "two non-authoritative project-native preflights mounted only on that selected stage, complete four-row pipeline and independently accepted two-kind build-exec plan",
        "two CPython and two project-native authority-bearing outputs plus complete per-run evidence",
        "native-rebuild evidence and CPython exclusion projection",
        "native-install, wheel, CPython-runtime, ELF, source, init, FD, FP, audit, dispatch, import and version ledgers",
        "BASE bundle",
        "in-memory discovery capsule, zero-process precommit and controller-input construction",
        "durable discovery request atomically commits and reopens those exact nested bytes while process count remains zero, then operational memfds/launch excluding FD7",
        "FD7 carrier and independent eight-stage evidence",
        "twin child streams and controller joins",
        "discovery evidence/common output/root",
        "trusted-generated index binding BASE",
        "FINAL bundle",
        "final CPU environment binding FINAL",
        "for each of the first 54 receipt-roster owners after its frozen inputs exist, one full request with operational memfds/launch excluding FD7 and then its FD7 carrier",
        "all 54 pre-G5a child transcript/result/EOF/waitpid joins and controller durable final traces complete",
        "the exact 54-row pre-G5a executable-origin completion index",
        "one external G5A-COMMIT full request whose dispatch handler inside the child consumes that completion index and returns internally with the unchanged G5a root/capability result; the child then emits its terminal and EOF, the controller waitpid-joins and durably persists the excluded surrounding trace/attestation, and only after that rehash may the external controller request emit and return the result",
        "at evaluator start one external G5B-JOIN full request atomically consumes the grant, performs the exact vault checks and in that same nonresumable child invokes the unique internal K7-FINAL continuation exactly once with no second request",
        "after the G5B-JOIN handler and its K7-FINAL continuation return inside the child, the child terminal/EOF and controller waitpid join occur, their surrounding trace/attestation are durably persisted and rehashed, and only then may the external G5B controller request emit and return its result; G5A/G5B/K7 surrounding evidence never backfills the completion index, G5a root, capability or result"
      ]
    },
    "normative_precedence": {
      "addition_map": {
        "/cpu_replay_environment/kernel_runtime_evidence_sha256": "/round7_native_wave2_closure/audit_and_trace/dynamic_mapping_closure/kernel_runtime_evidence",
        "/cpu_replay_environment/native_rebuild_evidence_index_sha256": "/round7_native_wave2_closure/build_and_deploy/native_rebuild_evidence/index",
        "/cpu_replay_environment/runtime_ledger_bundle_sha256": "/round7_native_wave2_closure/ledger_bundles/final_environment/rule"
      },
      "addition_paths": [
        "/cpu_replay_environment/kernel_runtime_evidence_sha256",
        "/cpu_replay_environment/native_rebuild_evidence_index_sha256",
        "/cpu_replay_environment/runtime_ledger_bundle_sha256"
      ],
      "future_artifact_schema_targets": [
        "/cpu_replay_environment/runtime_launcher",
        "/executed_code_closure/completion_index"
      ],
      "preserved": "Every Round6 scientific, teacher, tune, score, natural-inference, K1, target, prediction, capability, F4, job, hour, data, threshold and claim rule remains normative unless a path is explicitly replaced below.",
      "replacement_map": {
        "/cpu_replay_environment/cpython_initialization_profile/call_sequence/1": "/round7_native_wave2_closure/transport_and_launch/cpython_profile_overlay/replacement_rows/0/value",
        "/cpu_replay_environment/cpython_initialization_profile/call_sequence/11": "/round7_native_wave2_closure/transport_and_launch/cpython_profile_overlay/replacement_rows/3/value",
        "/cpu_replay_environment/cpython_initialization_profile/call_sequence/2": "/round7_native_wave2_closure/transport_and_launch/cpython_profile_overlay/replacement_rows/1/value",
        "/cpu_replay_environment/cpython_initialization_profile/call_sequence/7": "/round7_native_wave2_closure/transport_and_launch/cpython_profile_overlay/replacement_rows/2/value",
        "/cpu_replay_environment/cpython_initialization_profile/error_handling": "/round7_native_wave2_closure/transport_and_launch/cpython_profile_overlay/replacement_rows/4/value",
        "/cpu_replay_environment/digest_binding": "/round7_native_wave2_closure/ledger_bundles/final_environment/rule",
        "/cpu_replay_environment/fd_contract": "/round7_native_wave2_closure/transport_and_launch/fd_contract_composition + /round7_native_wave2_closure/transport_and_launch/carrier + /round7_native_wave2_closure/transport_and_launch/fd_layouts + /round7_native_wave2_closure/transport_and_launch/fd_readback + /round7_native_wave2_closure/transport_and_launch/operational_memfd_construction + /round7_native_wave2_closure/transport_and_launch/launch_instance + /round7_native_wave2_closure/transport_and_launch/launch_evidence_index",
        "/cpu_replay_environment/floating_point_control/helper/origin": "/round7_native_wave2_closure/build_and_deploy/fp_native_abi/floating_point_profile_overlay/value",
        "/cpu_replay_environment/required_bindings": "/round7_native_wave2_closure/ledger_bundles/final_environment/preservation + /round7_native_wave2_closure/ledger_bundles/final_environment/rule",
        "/cpu_replay_environment/runtime_launcher": "/round7_native_wave2_closure/controller/runtime_launcher_artifact_schema",
        "/cpu_replay_environment/schema": "/round7_native_wave2_closure/ledger_bundles/final_environment/schema",
        "/cpu_replay_environment/top_level_closed_keys": "/round7_native_wave2_closure/ledger_bundles/final_environment/closed_keys",
        "/cpu_replay_environment/trusted_generated_source_index": "/round7_native_wave2_closure/generated_source_closure",
        "/executed_code_closure/completion_index": "/round7_native_wave2_closure/code_and_install_closure/completion_index",
        "/executed_code_closure/executable_origin_trace": "/round7_native_wave2_closure/audit_and_trace",
        "/executed_code_closure/invocation": "/round7_native_wave2_closure/dispatch_abi",
        "/executed_code_closure/noncircularity": "/round7_native_wave2_closure/ledger_bundles/noncircularity + /round7_native_wave2_closure/ledger_bundles/topological_order",
        "/executed_code_closure/source_allowlist": "/round7_native_wave2_closure/code_and_install_closure/source_allowlist"
      },
      "replacement_paths": [
        "/cpu_replay_environment/cpython_initialization_profile/call_sequence/1",
        "/cpu_replay_environment/cpython_initialization_profile/call_sequence/11",
        "/cpu_replay_environment/cpython_initialization_profile/call_sequence/2",
        "/cpu_replay_environment/cpython_initialization_profile/call_sequence/7",
        "/cpu_replay_environment/cpython_initialization_profile/error_handling",
        "/cpu_replay_environment/digest_binding",
        "/cpu_replay_environment/fd_contract",
        "/cpu_replay_environment/floating_point_control/helper/origin",
        "/cpu_replay_environment/required_bindings",
        "/cpu_replay_environment/runtime_launcher",
        "/cpu_replay_environment/schema",
        "/cpu_replay_environment/top_level_closed_keys",
        "/cpu_replay_environment/trusted_generated_source_index",
        "/executed_code_closure/completion_index",
        "/executed_code_closure/executable_origin_trace",
        "/executed_code_closure/invocation",
        "/executed_code_closure/noncircularity",
        "/executed_code_closure/source_allowlist"
      ],
      "rule": "For exactly replacement_paths, each left-hand JSON pointer must resolve in the accepted Round6 object. For the two and only two paths in future_artifact_schema_targets, the mapped Round7 object replaces the old value's construction semantics and closed schema, not the literal future value: /cpu_replay_environment/runtime_launcher is the future 18-key instance governed by runtime_launcher_artifact_schema closed_keys/static_values/future_fields, and /executed_code_closure/completion_index is the future 54-row index governed by completion_index; neither schema descriptor may itself be substituted as the runtime artifact. For every other replacement path, the complete mapped Round7 object/value or explicitly listed composition replaces the existing value literally; mixing fields or fallback is forbidden. For exactly addition_paths, the left-hand pointer must be absent from Round6 and is added once to the v5 final environment with the complete mapped semantics; treating it as a replacement, omission or duplicate fails. All other Round6 fields remain unchanged. The Round6 prose statement that a discovery five-frame stream included launch and omitted configuration was erroneous: its JSON actually specified three FD readbacks, configuration and output; Round7 replaces both forms with the explicit v5 stream.",
      "status": "PROSPECTIVE_UNTIL_FRESH_ROUND7_ACCEPT"
    },
    "required_round7_tests": {
      "negative": [
        "missing, altered, wrong-FD, offset-drifted or unsealed FD7 carrier; FD7 included in launch rows/index; launch bytes or hash mismatch; operational memfd/readback stage omission, wrong close schedule, O_NOFOLLOW procfs reopen, status-flag drift or extra descriptor",
        "phase-1 failure with nonnull launch hash or nonzero mode, LAUNCH_HASH failure without the exact parsed mode or with a nonnull digest, any failure after successful LAUNCH_HASH with null/wrong digest, truncated/reordered/duplicate/trailing frame, missing EOF, wrong exit, child stderr/stdout byte, broken stream and exec/signal synthesis",
        "wrong controller/child argv or image, Python controller, wrong env/cwd/uid/gid/umask/FD action, Round7 selection of the old stdlib-only discovery PyConfig map, profile-overlay pointer/value drift, parallel twin launch, retry, non-durable success, overwrite/EEXIST mismatch or self-referential attestation",
        "bundle missing/extra/reordered/substituted row, count/hex/hash mismatch, FINAL prefix drift, environment inside bundle, trusted row binding FINAL, launch/bundle/readback mismatch or backedge",
        "dispatch non-bytes/subclass/second arg/keyword, wrong owner or handler, input hex/hash/schema mismatch, non-bytes/oversize/noncanonical result, leaked Python exception, K7 external invocation/twice/wrong process, forbidden handler side effect",
        "3.12.13 science relabel, 3.12.4 supervisor substitution, historical X0/operator foreign-key rewrite or cross-layer result replacement",
        "missing/tampered/reordered project or third-party generated row, twin mismatch/empty output, native/no-frame generator, delayed or unmatched compile-exec, generated action after DISPATCH_BEGIN or discovery-learned BASE row",
        "compiler/toolchain/header/library/build-image/hash mismatch, missing/extra/reordered build-exec row, changed flag, fast-math/LTO/strip/shell/python-config, missing or separately installed native FP object, compile-mode/ABI/readback drift, nonidentical rebuild, editable/wheel member mismatch, _fp_control false wheel/repository mapping, wrong installed path or installed extension mislabeled as wheel association",
        "toolchain ledger containing a run-output digest, missing/unretrievable accepted exec plan, CPython/project run-kind swap, generated executable without same-run producer/consumer evidence, mixed-run output index, missing pre-run input snapshot, input mount mutation, per-command stream concatenation, -MMD/system-header omission, CPython stage/exclusion prefix mismatch or nonidentical twin stage",
        "outer or child exec environment absent/reordered/unreviewed, nondeterministic secret/clock/random/PID env, writable input mount, wrong extraction mount transition, wrong uid/gid/mode/owner, missing temp/cache/home member, incomplete build/output/intermediate roster, third-party wheel omission, native-install uid/gid drift or deploy prefix alias",
        "controller/stage/dispatch/discovery complete-object cap exceeded at any hex layer, oversized single frame, ledger/request cap misapplied to decoded bytes, one-shot write above MAX_RW_COUNT, wrong chunk offset/retry/short-write behavior or size overflow",
        "missing/extra/reordered initialization audit prefix, initialization raw attributed to a top-level import, dynamic group learned from runtime, group ordinal gap, primary/ELF array mismatch, one per-loader delta, shared dependency guessed onto one primary, extension/ctypes event after DISPATCH_BEGIN, unframed mapping delta, VDSO/vsyscall drift or unaccounted executable mapping",
        "unknown raw audit name/arity/type, provider classified before barrier, eval guessed from exec, raw repr/pointer, compile_mode field, dangerous code/marshal/ctypes action, missing/extra normalized event, omitted/reordered Py_FinalizeEx raw shutdown event, post-output executable origin, child-recorded join or join as token 17"
      ],
      "positive": [
        "two distinct discovery children with exact FD6/7/8 transport produce byte-identical nonempty outputs and externally joined evidence",
        "one full child validates FINAL plus separate environment, consumes all trusted rows, dispatches the exact bytes ABI, reaches G5B-to-K7 continuation, and emits a durable controller success only after EOF/waitpid/rehash",
        "57 descriptors mechanically constructed from exact owner order/defaults/overrides and kept separate from exact 54-owner/106-edge receipt projection",
        "project installed bytes equal repository allowlist and wheel member; native outputs resolve through build/install/ELF ledgers; every raw executable event resolves through the closed catalog",
        "four build preflights yield one reviewed run-independent exec plan; two CPython and two project-native authority runs reproduce exact exec environments, generated-image provenance, input snapshots, per-command streams, -MD dependencies, output/temp ledgers and twin equality before native-install",
        "the exact initialization prefix and every explicit top-level import group reproduce precommitted raw/normalized rows and at most one collective executable-mapping delta; ordered replay including kernel mappings equals the fresh terminal mapping set with zero post-dispatch load"
      ],
      "status": "SPECIFIED_NOT_RUN"
    },
    "round7_closure_matrix": [
      {
        "closure": "EXACTLY_SPECIFIED",
        "id": "R7-B1",
        "rule": "FD7 carries complete launch bytes/hash before any CPython API, is independently evidenced and excluded from launch; operational memfd/readback transitions are closed; FD8 carries exact BASE/FINAL ledgers and actual child readback equals controller construction."
      },
      {
        "closure": "EXACTLY_SPECIFIED",
        "id": "R7-B2",
        "rule": "Universal FD6 and exact frame/error/EOF/exit rules carry the three-key native failure record from the earliest child boundary; controller synthesizes only explicitly unobservable exec/signal/broken-stream failures."
      },
      {
        "closure": "EXACTLY_SPECIFIED",
        "id": "R7-B3",
        "rule": "One source/ELF owns exact controller and child modes, request/argv/env/FD/fork order, nonretry attempt marker, prelaunch/trace paths and durable no-overwrite publication; controller alone persists and joins."
      },
      {
        "closure": "EXACTLY_SPECIFIED",
        "id": "R7-B4",
        "rule": "Exact byte-carrying BASE/FINAL bundle schemas, role orders, separate final environment carrier and topological exclusions remove ledger/path and self-cycle ambiguity."
      },
      {
        "closure": "EXACTLY_SPECIFIED",
        "id": "R7-B5",
        "rule": "Literal bytes-only Python ABI, mechanical 57-row descriptor construction, result wrappers, native call sequence, exception mapping and side-effect profiles freeze dispatch without implementing it."
      },
      {
        "closure": "EXACTLY_SPECIFIED",
        "id": "R7-B6",
        "rule": "3.12.4 historical science and stock 3.12.13 provenance verifier are separate typed version-layer rows; verifier evidence cannot replace historical artifacts."
      },
      {
        "closure": "EXACTLY_SPECIFIED_FAIL_CLOSED",
        "id": "R7-B7",
        "rule": "BASE-precommitted full project/third-party import plan plus twin nonempty discovery and exact full-process consumption covers actual dynamic generation; unobservable/delayed/nondeterministic generation fails instead of being guessed."
      },
      {
        "closure": "EXACTLY_SPECIFIED_FUTURE_VALUES_REQUIRED",
        "id": "R7-B8",
        "rule": "Pure pre-build toolchain ledger, four preflight instances, independently accepted two-kind exec plan, four authority-bearing run evidence rows, CPython/project-native twin output/input/temp/stream equality, generated-exec provenance, exact -MD closure, build/install mappings, dual-mode FP ABI and no-wheel associations are frozen; future byte digests must be populated/reviewed before P00/P06, never fabricated."
      },
      {
        "closure": "EXACTLY_SPECIFIED",
        "id": "R7-B9",
        "rule": "Closed raw catalog, stock arities, initialization plus top-level-import grouped safe barriers, zero eval mapping, thirteen typed origins, exact dynamic mapping groups, evidence-finalized controller PROCESS_EXEC, complete Py_FinalizeEx suffix, typed raw/normalized rows and controller-only joins uniquely map executable origins to the preserved 16 tokens."
      }
    ],
    "scope_and_receipt_projection": {
      "authority": "All authority bits remain zero. Native build, discovery, dispatch, trace, G5a, capability, K7 and experiment execution remain future and separately authorized.",
      "current_implementation_boundary": "The accepted Wave1 postfix implementation validates only the evidence-backed exact 54-node/106-edge pre-G5a projection and explicitly has no generic/full-DAG mode. Round7 specifies future closure but does not claim it exists in current code.",
      "dispatch_vs_receipts": "The 57 dispatch descriptors are not receipt-DAG nodes. The exact 54 roster owners and their 106 direct role edges are a pre-G5a projection inside a future full receipt DAG, not a total-node or total-edge bound on that full DAG.",
      "full_dag_rule": "Future G5a construction derives the complete expected node/edge multiset from exact receipt bytes and typed artifact/index rows, then requires equality. The 54 roster nodes must all exist with typed owner equality; filtering their roster-direct roles stage.upstream,run.upstream,g1.teacher-run,k1.upstream,natp.k6 must reproduce exactly 106. Legitimate feature/checkpoint/evaluation/selection/target/prediction edges are additional and independently exact.",
      "method_lock": "One dominant method, 27 jobs, 93 A6000 hours, 402 identities, 9,648 prediction envelopes, 134 development identities, K1 floors, seven-member target, ten-key target receipt, G5a key set, thresholds, data boundary, evaluator order and paper claims are unchanged."
    },
    "transport_and_launch": {
      "carrier": {
        "child_algorithm": "At physical entry snapshot all descriptors without opening project or CPython bytes; pread FD7 from offset zero, verify size/EOF/canonical carrier/hash without moving its offset, snapshot carrier state, close FD7 and prove F_GETFD=-1/EBADF. Only then parse launch, validate FD8 and emit the carrier-readback frame. No CPython API occurs first.",
        "closed_keys": [
          "launch_instance_bytes",
          "launch_instance_bytes_hex",
          "launch_instance_sha256",
          "schema"
        ],
        "construction": "FD7 is a sealed read-only memfd built with the exact eight-stage writer-open/seal/distinct-OFD algorithm below. Its construction evidence is external to launch and compared by the controller with child carrier readback after join.",
        "construction_evidence": {
          "closed_keys": [
            "carrier_bytes",
            "carrier_sha256",
            "reader_fd",
            "rows",
            "schema",
            "writer_fd"
          ],
          "digest_rule": "carrier_construction_sha256 is SHA-256 of the complete standalone canonical construction-evidence bytes including LF. The digest is external and absent from the object. carrier_bytes is the positive exact payload count and carrier_sha256 hashes the same complete carrier bytes named by the controller request/launch evidence.",
          "row_closed_keys": [
            "child_descriptor_flags_u32_or_null",
            "child_offset_or_null",
            "child_status_flags_u32_or_null",
            "reader_descriptor_flags_u32_or_null",
            "reader_offset_or_null",
            "reader_status_flags_u32_or_null",
            "same_device_or_null",
            "same_inode_or_null",
            "same_open_file_description_or_null",
            "seals",
            "size_bytes",
            "stage",
            "writer_descriptor_flags_u32_or_null",
            "writer_offset_or_null",
            "writer_status_flags_u32_or_null"
          ],
          "row_order": [
            "CREATE_WRITER",
            "WRITE",
            "VERIFY_WRITER",
            "SEAL_WRITER_OPEN",
            "OPEN_READER",
            "VERIFY_READER",
            "CLOSE_WRITER",
            "INSTALL_TARGET"
          ],
          "row_rule": "Exactly eight rows in row_order. The containing launch-evidence row fixes writer_fd/reader_fd to 3/4 for either discovery mode or 7/9 for full-runtime, exactly as controller.child_fd_actions.scratch_rule derives; no other integers are valid. size_bytes always equals carrier_bytes. CREATE_WRITER has writer FD_CLOEXEC/O_RDWR|O_LARGEFILE/offset0, all reader/child/same fields null and seals []; WRITE and VERIFY_WRITER retain writer flags/status with offset=carrier_bytes, reader/child/same fields null and seals []; SEAL_WRITER_OPEN is the same with the exact four seals; OPEN_READER has writer offset=carrier_bytes and reader FD_CLOEXEC/O_RDONLY|O_LARGEFILE/offset0, child/same fields null and exact seals; VERIFY_READER has the same fd values and same_device=true,same_inode=true,same_open_file_description=false; CLOSE_WRITER has writer fields null after EBADF, reader unchanged and the same three booleans retained; INSTALL_TARGET has writer/reader fields null, child flags/status/offset=0,O_RDONLY|O_LARGEFILE,0, exact seals and same_device=true,same_inode=true,same_open_file_description=true against the immediately preceding reader. Every stage also proves the exact bytes/hash/EOF and ordered negative operations required by construction_stages.",
          "schema": "temporac.native-launch-carrier-construction.v5"
        },
        "construction_stages": [
          "1 CREATE_WRITER: memfd_create exact name temporac-launch-carrier-v5 and flags MFD_CLOEXEC|MFD_ALLOW_SEALING; require fresh fd>=3, F_GETFL exactly O_RDWR|O_LARGEFILE with no append/nonblock/async/direct/sync/path bits, F_GETFD=FD_CLOEXEC, regular anonymous memfd, size=0, offset=0 and seals=0.",
          "2 WRITE: from the canonical in-memory carrier bytes, use an EINTR-safe write loop on the writer until the positive exact byte count is written; zero progress or any short terminal write fails; require writer offset=count and fstat size=count.",
          "3 VERIFY_WRITER: pread from offset zero without changing writer offset, require complete bytes, next-byte EOF, exact SHA-256 and fstat identity/size; no mmap, lseek-dependent read, truncate or second writer exists.",
          "4 SEAL_WRITER_OPEN: while the sole O_RDWR writer remains open, F_ADD_SEALS exactly F_SEAL_SEAL|F_SEAL_SHRINK|F_SEAL_GROW|F_SEAL_WRITE in one call; F_GET_SEALS must equal that mask and a one-byte pwrite plus ftruncate(size+1) must each fail EPERM without changing bytes, size or offset.",
          "5 OPEN_READER: open exact /proc/self/fd/<unsigned-decimal-writer-fd> with O_RDONLY|O_CLOEXEC and no mode argument. The procfs final component is necessarily a symlink, so O_NOFOLLOW is forbidden here; before open require /proc is the precommitted procfs mount and the decimal token is the live writer integer. Require returned reader_fd=4 when discovery writer_fd=3 or reader_fd=9 when full writer_fd=7, a distinct fd and distinct open-file description proven by setting reader offset=0 while writer offset remains count, then exact st_dev/st_ino/type equality prevents substitution.",
          "6 VERIFY_READER: require reader F_GETFL exactly O_RDONLY|O_LARGEFILE with no append/nonblock/async/direct/sync/path bit, FD_CLOEXEC, same st_dev/st_ino/type/size and exact seals, offset=0, complete pread bytes/hash/EOF, and no offset movement.",
          "7 CLOSE_WRITER: close the writer exactly once; F_GETFD on that integer must fail EBADF while reader identity, seals, offset, size, bytes, hash and EOF remain unchanged.",
          "8 INSTALL_TARGET: after CLOSE_WRITER, call dup3(reader_fd,7,0) exactly once, require success at FD7, close reader_fd and prove EBADF. The exact derived reader is never 7, so clearing CLOEXEC in place or any alternate branch is forbidden. Require only FD7 remains, F_GETFD=0 so the immediately following single child exec inherits it, and the exact read-only state, offset=0, identity, seals, count, bytes, hash and EOF. The controller forks no unrelated process, closes its parent FD7 immediately after the one fork, and the embedded child sets CLOEXEC before any possible later exec. Any collision with an operational fd, retry, leaked alias or extra descriptor fails."
        ],
        "cycle_rule": "FD7 and its payload/hash/construction evidence are absent from launch_instance.fd_rows and the launch sealed-memfd construction index. Launch therefore cannot hash the carrier that contains launch bytes.",
        "encoding_rule": "launch_instance_bytes is the positive exact byte count; launch_instance_bytes_hex is lowercase complete launch bytes of exactly that count; launch_instance_sha256 hashes those bytes; strict parse/re-encode must be byte-identical.",
        "readback": {
          "closed_keys": [
            "carrier_bytes",
            "carrier_sha256",
            "extra_fds",
            "fd",
            "launch_instance_sha256",
            "operational_rows",
            "rlimit_nofile_cur",
            "rlimit_nofile_max",
            "schema",
            "stage",
            "transport_carrier_row"
          ],
          "operational_row_closed_keys": [
            "access_mode_or_null",
            "descriptor_flags_u32_or_null",
            "device_major_or_null",
            "device_minor_or_null",
            "fd",
            "file_status_flags_u32_or_null",
            "inheritable_or_null",
            "object_type_or_null",
            "offset_or_null",
            "pipe_capacity_bytes_or_null",
            "queued_bytes_or_null",
            "seals_or_null",
            "seekable_or_null",
            "size_bytes_or_null",
            "state"
          ],
          "operational_rows_rule": "At physical entry, operational_rows contains exactly the applicable mode's operational fd skeleton in ascending fd order, every row has exactly operational_row_closed_keys, state=present, descriptor flags zero, and every fresh OS value equals the launch row under the preserved object-type rules. FD7 is excluded from this array and represented only by transport_carrier_row; extra_fds is exactly []. rlimit_nofile_cur and rlimit_nofile_max both equal 256 and the fcntl scan covers every integer 0..255 without opening a scan descriptor.",
          "rule": "fd=7 and stage=carrier-validated-and-closed. carrier_bytes/carrier_sha256 equal controller construction, launch_instance_sha256 equals the parsed carrier, operational_rows/extra_fds/rlimits satisfy operational_rows_rule, and transport_carrier_row is the exact physical-entry FD7 projection captured before close plus closed_after_validation=true and post_close_errno=EBADF. The controller compares every comparable construction/readback field after waitpid; mismatch fails evidence.",
          "schema": "temporac.native-launch-carrier-readback.v5",
          "transport_carrier_row_closed_keys": [
            "access_mode",
            "closed_after_validation",
            "descriptor_flags_entry_u32",
            "descriptor_flags_locked_u32",
            "device_major_or_null",
            "device_minor_or_null",
            "eof_rule",
            "fd",
            "file_status_flags_u32",
            "inheritable_after_lock",
            "inheritable_at_exec_entry",
            "initial_offset_or_null",
            "object_type",
            "payload_bytes_or_null",
            "payload_sha256_or_null",
            "pipe_capacity_bytes_or_null",
            "post_close_errno",
            "queued_bytes_or_null",
            "role",
            "seals",
            "seekable",
            "size_bytes_or_null"
          ],
          "transport_carrier_row_rule": "The exact values are access_mode=read-only, fd=7, object_type=sealed-memfd, role=transport-carrier, descriptor_flags_entry_u32=0, inheritable_at_exec_entry=true, descriptor_flags_locked_u32=FD_CLOEXEC, inheritable_after_lock=false, file_status_flags_u32=O_RDONLY|O_LARGEFILE, seekable=true, initial_offset_or_null=0, size/payload count=carrier_bytes, payload digest=carrier_sha256, seals exactly the four ASCII-ordered seal tokens, eof_rule=sealed-exact-size-then-eof, all device/pipe fields null, closed_after_validation=true and post_close_errno=EBADF. The child sets FD_CLOEXEC immediately after the physical entry snapshot, before reading the payload; no exec can intervene."
        },
        "schema": "temporac.native-launch-carrier.v5"
      },
      "cpython_profile_overlay": {
        "artifact_rule": "Construct the BASE cpython-initialization-profile role by taking the exact accepted Round6 temporac.cpython-initialization-profile.v4 object and replacing only the five JSON pointers in replacement_rows with their complete listed values. All constructor defaults, 10 PyPreConfig fields, 64 PyConfig fields, owned-string/list setters, source bindings, readback schemas, field orders and every other field remain byte-for-byte normative. Serialize the resulting complete top-level object canonically with LF and hash it as cpython_initialization_profile_sha256. Literal old-object hashing or partial prose substitution fails.",
        "replacement_row_closed_keys": [
          "json_pointer",
          "value"
        ],
        "replacement_rows": [
          {
            "json_pointer": "/cpu_replay_environment/cpython_initialization_profile/call_sequence/1",
            "value": "Still before any CPython API, execute the exact v5 carrier close and operational FD transition: set FD_CLOEXEC on the mode-specific operational descriptors; pread/verify and close the exact sealed inputs; require retained/closed states and no extras; and commit locked-pre-cpython readback as specified by transport_and_launch.fd_readback."
          },
          {
            "json_pointer": "/cpu_replay_environment/cpython_initialization_profile/call_sequence/2",
            "value": "Still before any CPython API and immediately after locked-pre-cpython readback, execute build_and_deploy.fp_native_abi.call_order through the statically linked hidden ABI: require set_round(FE_TONEAREST), set_mxcsr(0x00001f80), get_round and get_mxcsr exact success/readback, with failure mapped only to FP_LOCK. Then perform exactly audit_and_trace.audit_buffer.allocation and its protection/readback with failure AUDIT_BUFFER_ALLOC. Then call PySys_AddAuditHook exactly once and require return 0; then call PyPreConfig_InitIsolatedConfig(&preconfig) exactly once. No other CPython API, explicit PyMem call, locale conversion, initialization call or floating-point operation occurs before or between those ordered boundaries."
          },
          {
            "json_pointer": "/cpu_replay_environment/cpython_initialization_profile/call_sequence/7",
            "value": "The supervisor does not call PyConfig_Read. Serialize the complete native pre-initialize struct snapshot in header order. Every Round7 discovery-run and full-runtime child selects the exact preserved pyconfig_full_runtime_final_values map; the preserved pyconfig_discovery_final_values map remains an unselected historical Round6 value and cannot be used for Round7 evidence."
          },
          {
            "json_pointer": "/cpu_replay_environment/cpython_initialization_profile/call_sequence/11",
            "value": "After clear, every mode executes the exact BASE pre-dispatch import plan through complete trusted-generation handling. A discovery run then serializes its non-authorizing output without dispatch; full runtime consumes the committed trusted sequence, emits DISPATCH_BEGIN and invokes exact byte dispatch. Joined normal exit calls Py_FinalizeEx exactly once while the audit hook frames every shutdown event; nonzero, unframed shutdown event or new executable origin fails evidence."
          },
          {
            "json_pointer": "/cpu_replay_environment/cpython_initialization_profile/error_handling",
            "value": {
              "error_exit_code": 120,
              "error_record": {
                "closed_keys": [
                  "launch_instance_sha256_or_null",
                  "schema",
                  "stage"
                ],
                "rule": "Use exactly temporac.native-failure.v5 and the universal FD6 child error frame. Every FD/FP/CPython profile-stage failure represented by this local object occurs after successful LAUNCH_HASH and therefore repeats the exact nonnull launch digest. No CPython message, path, errno, exception, traceback or alternate field is serialized.",
                "schema": "temporac.native-failure.v5"
              },
              "rule": "Every PyStatus is tested with PyStatus_Exception immediately; every void constructor/clear/native operation governed by this profile handler is followed by its specified structural/readback check. The first represented failing boundary maps to exactly one stage token below, calls PyConfig_Clear exactly once iff the preserved lifecycle requires it, emits one v5 failure frame, closes authority FDs and _exit(120). DISPATCH and RESULT are deliberately absent from this local list and are handled only by the outer framing_and_failures state machine with exit121; after a successful RESULT, FINALIZE,TERMINAL_MAPPING_CAPTURE and AUDIT_BUFFER_RELEASE reenter this local exit120 domain. Do not retry, fall back, call Py_ExitStatusException, finalize a partial state, reuse it, or emit the superseded v4 initialization-error object.",
              "stage_tokens": [
                "INITIAL_MAPPING_CAPTURE",
                "FD_EXEC_ENTRY",
                "FD_LOCK",
                "FD_LOCKED_READBACK",
                "FP_LOCK",
                "AUDIT_BUFFER_ALLOC",
                "AUDIT_HOOK",
                "AUDIT_BUFFER_CAPTURE",
                "PY_PREINITIALIZE",
                "PY_CONFIG",
                "PY_INITIALIZE",
                "INITIALIZATION_MAPPING_CAPTURE",
                "POST_INITIALIZE_READBACK",
                "AUDIT_BUFFER_FLUSH",
                "PRE_DISPATCH_IMPORT",
                "IMPORT_GROUP_MAPPING_CAPTURE",
                "TRUSTED_GENERATED_CONSUMPTION",
                "FINALIZE",
                "TERMINAL_MAPPING_CAPTURE",
                "AUDIT_BUFFER_RELEASE"
              ]
            }
          }
        ],
        "replacement_rule": "Rows are exactly replacement_rows order and their json_pointer values are unique. The first four values are exact JSON strings; the fifth is the exact closed error-handling object. The accepted Round7 successor and generated profile artifact commit these values before BASE; runtime does not patch or select text dynamically.",
        "selection_rule": "All three Round7 child modes deliberately select pyconfig_full_runtime_final_values because that already accepted 64-field map is the only preserved map whose module_search_paths includes exact site-packages required by the BASE import plan. Discovery is still distinguished only by launch mode, null environment, BASE bundle, FD layout and no-dispatch rule. Selecting the old stdlib-only discovery map, adding a path, using site.py/environment discovery, or inferring mode from PyConfig argv fails."
      },
      "fd_contract_composition": {
        "artifact": {
          "closed_keys": [
            "contract_sha256",
            "rows",
            "schema"
          ],
          "digest_rule": "fd_contract_sha256 is SHA-256 of the complete canonical artifact bytes including LF. The digest is external and absent from the object.",
          "row_closed_keys": [
            "json_pointer",
            "value_sha256"
          ],
          "row_order": [
            "/cpu_replay_environment/fd_contract/constant_ledger",
            "/cpu_replay_environment/fd_contract/fd_row_closed_keys",
            "/cpu_replay_environment/fd_contract/row_semantics/0",
            "/cpu_replay_environment/fd_contract/row_semantics/2",
            "/cpu_replay_environment/fd_contract/row_semantics/3",
            "/cpu_replay_environment/fd_contract/row_semantics/5",
            "/cpu_replay_environment/fd_contract/value_domains",
            "/round7_native_wave2_closure/transport_and_launch/carrier",
            "/round7_native_wave2_closure/transport_and_launch/fd_layouts",
            "/round7_native_wave2_closure/transport_and_launch/fd_readback",
            "/round7_native_wave2_closure/transport_and_launch/launch_evidence_index",
            "/round7_native_wave2_closure/transport_and_launch/launch_instance",
            "/round7_native_wave2_closure/transport_and_launch/operational_memfd_construction"
          ],
          "row_rule": "Resolve each pointer in the exact accepted timestamped Amendment005 JSON, serialize that complete JSON value alone by canonical compact recursively sorted-key UTF-8 JSON plus one LF, and set value_sha256 to SHA-256 of those bytes. Rows are exactly row_order, with no mutable fixed alias, missing/extra pointer, copied prose or alternate encoding. contract_sha256 is the accepted Round7 successor digest.",
          "schema": "temporac.linux-fd-contract.v5"
        },
        "effective_schema": "temporac.linux-fd-contract.v5",
        "preservation_rule": "Each preserved pointer resolves inside this same Amendment005 JSON to the exact accepted Round6 value; that complete value remains normative byte-for-byte. In the effective v5 composition, the preserved row_semantics/0 phrase 'the first native snapshot' denotes specifically the first temporac.linux-fd-readback.v5 operational snapshot at stage=post-carrier-exec-entry. The earlier temporac.native-launch-carrier-readback.v5 physical-entry scan is a separate preliminary transport snapshot: immediately afterward only FD7 is locked/read/closed under its dedicated rule, while all operational descriptors retain flags zero through post-carrier-exec-entry; only after that operational readback does preserved row_semantics/0 lock them. This contextual specialization is unique and prevents the transport scan from silently triggering the operational lock schedule. No prose summary, mutable alias, implementation default or partial copy may substitute. All other Round6 fd-contract members are replaced by the named v5 objects below, so old discovery FD3, old layouts, old close schedule, old memfd applicability/name and old launch/readback schema tokens have no residual authority.",
        "preserved_round6_paths": [
          "/cpu_replay_environment/fd_contract/constant_ledger",
          "/cpu_replay_environment/fd_contract/fd_row_closed_keys",
          "/cpu_replay_environment/fd_contract/row_semantics/0",
          "/cpu_replay_environment/fd_contract/row_semantics/2",
          "/cpu_replay_environment/fd_contract/row_semantics/3",
          "/cpu_replay_environment/fd_contract/row_semantics/5",
          "/cpu_replay_environment/fd_contract/value_domains"
        ],
        "replaced_round6_paths": [
          "/cpu_replay_environment/fd_contract/discovery_layout",
          "/cpu_replay_environment/fd_contract/full_runtime_layout",
          "/cpu_replay_environment/fd_contract/launch_instance",
          "/cpu_replay_environment/fd_contract/readback",
          "/cpu_replay_environment/fd_contract/row_semantics/1",
          "/cpu_replay_environment/fd_contract/row_semantics/4",
          "/cpu_replay_environment/fd_contract/schema",
          "/cpu_replay_environment/fd_contract/sealed_memfd_construction"
        ],
        "replacement_rule": "The effective v5 contract is exactly the preserved values plus transport_and_launch.carrier, fd_layouts, operational_memfd_construction, fd_readback, launch_instance and launch_evidence_index. A validator rejects any duplicate semantic source, any inherited Round6 value not explicitly preserved, or any disagreement between a preserved value and its v5 use."
      },
      "fd_layouts": {
        "discovery": [
          "fd0 sealed-ro discovery-controller input payload",
          "fd1 independent write-only /dev/null",
          "fd2 independent write-only /dev/null",
          "fd6 universal child event/error write pipe",
          "fd7 transport-only launch carrier, closed before operational readback",
          "fd8 sealed-ro BASE runtime-ledger bundle"
        ],
        "discovery_memfd_construction_index": [
          0,
          8
        ],
        "discovery_operational_rows": [
          {
            "access_mode": "read-only",
            "fd": 0,
            "object_type": "sealed-memfd",
            "role": "discovery-controller-input"
          },
          {
            "access_mode": "write-only",
            "fd": 1,
            "object_type": "char-device-1:3",
            "role": "stdout-null"
          },
          {
            "access_mode": "write-only",
            "fd": 2,
            "object_type": "char-device-1:3",
            "role": "stderr-null"
          },
          {
            "access_mode": "write-only",
            "fd": 6,
            "object_type": "pipe-write-end",
            "role": "child-event-output"
          },
          {
            "access_mode": "read-only",
            "fd": 8,
            "object_type": "sealed-memfd",
            "role": "base-runtime-ledger-bundle"
          }
        ],
        "full": [
          "fd0 sealed-ro temporac.invocation.v4",
          "fd1 independent write-only /dev/null",
          "fd2 independent write-only /dev/null",
          "fd3 sealed-ro accepted effective-contract index",
          "fd4 sealed-ro source allowlist",
          "fd5 sealed-ro final CPU environment",
          "fd6 universal child event/error write pipe",
          "fd7 transport-only launch carrier, closed before operational readback",
          "fd8 sealed-ro FINAL runtime-ledger bundle"
        ],
        "full_memfd_construction_index": [
          0,
          3,
          4,
          5,
          8
        ],
        "full_operational_rows": [
          {
            "access_mode": "read-only",
            "fd": 0,
            "object_type": "sealed-memfd",
            "role": "invocation"
          },
          {
            "access_mode": "write-only",
            "fd": 1,
            "object_type": "char-device-1:3",
            "role": "stdout-null"
          },
          {
            "access_mode": "write-only",
            "fd": 2,
            "object_type": "char-device-1:3",
            "role": "stderr-null"
          },
          {
            "access_mode": "read-only",
            "fd": 3,
            "object_type": "sealed-memfd",
            "role": "effective-contract"
          },
          {
            "access_mode": "read-only",
            "fd": 4,
            "object_type": "sealed-memfd",
            "role": "source-allowlist"
          },
          {
            "access_mode": "read-only",
            "fd": 5,
            "object_type": "sealed-memfd",
            "role": "cpu-environment"
          },
          {
            "access_mode": "write-only",
            "fd": 6,
            "object_type": "pipe-write-end",
            "role": "child-event-output"
          },
          {
            "access_mode": "read-only",
            "fd": 8,
            "object_type": "sealed-memfd",
            "role": "final-runtime-ledger-bundle"
          }
        ],
        "layout_rule": "The four-key operational skeleton rows above are authoritative and ordered by fd. Each launch fd_rows value is the exact twenty-key Round6 launch-FD row whose fd/access_mode/object_type/role projection equals the applicable skeleton. Sealed rows bind the exact named request/invocation, contract, allowlist, environment or bundle bytes; null and pipe rows use the preserved value domains. FD7 is never an operational row or memfd-index member.",
        "universal_stream_reason": "The same child argv cannot know discovery/full mode until it validates FD7; therefore FD6 is the stream in both modes and can carry a pre-carrier null-hash failure. Round6 discovery FD3 stream is superseded."
      },
      "fd_readback": {
        "closed_keys": [
          "extra_fds",
          "launch_instance_sha256",
          "rlimit_nofile_cur",
          "rlimit_nofile_max",
          "rows",
          "schema",
          "stage"
        ],
        "extra_rule": "The physical-entry snapshot is serialized only by the closed temporac.native-launch-carrier-readback.v5 object as its operational_rows plus one transport_carrier_row and empty extra_fds. After verified FD7 close, each ordinary linux-fd-readback rows array uses exactly the mode layout without FD7 and extra_fds is empty. No undefined transport field or uncommitted physical descriptor exists.",
        "index": {
          "closed_keys": [
            "launch_instance_sha256",
            "readbacks",
            "schema"
          ],
          "digest_rule": "operational_fd_readback_index_sha256 is SHA-256 of the exact standalone canonical index bytes including LF; the digest is external and absent from the index.",
          "readback_order": [
            "post-carrier-exec-entry",
            "locked-pre-cpython",
            "post-pyinitialize"
          ],
          "row_closed_keys": [
            "readback_bytes_hex",
            "readback_sha256",
            "stage"
          ],
          "row_rule": "Exactly three rows in readback_order. readback_bytes_hex is lowercase complete temporac.linux-fd-readback.v5 canonical bytes including LF, hashes exactly, and embeds the repeated launch digest and stage. Missing, extra, duplicate, reordered, digest-only or cross-launch evidence fails.",
          "schema": "temporac.linux-fd-readback-index.v5"
        },
        "physical_enumeration": "Controller and child first require the exact committed RLIMIT_NOFILE cur=max=256. Iterate every integer fd 0 through 255 in ascending order using only fcntl(fd,F_GETFD). Record open rows for success and require errno=EBADF for every absent integer; EINTR is retried without advancing. This scan opens no `/proc/self/fd` directory and creates no self-observation fd. At child physical entry the exact carrier readback records applicable operational rows plus FD7 as transport_carrier_row and no other open integer. After FD7 close and every later mandated close, the same 256-integer scan proves the exact operational set and empty extras. Any rlimit drift fails.",
        "row_closed_keys": [
          "access_mode_or_null",
          "descriptor_flags_u32_or_null",
          "device_major_or_null",
          "device_minor_or_null",
          "fd",
          "file_status_flags_u32_or_null",
          "inheritable_or_null",
          "object_type_or_null",
          "offset_or_null",
          "pipe_capacity_bytes_or_null",
          "queued_bytes_or_null",
          "seals_or_null",
          "seekable_or_null",
          "size_bytes_or_null",
          "state"
        ],
        "row_rule": "At every operational stage rows cover every fd in the applicable operational skeleton, in ascending order, including rows for descriptors mandated closed. rlimit_nofile_cur=rlimit_nofile_max=256. state is exactly present or closed. A present row repeats fresh fstat/F_GETFD/F_GETFL/lseek-or-ESPIPE/F_GET_SEALS/F_GETPIPE_SZ/FIONREAD values and equals the launch row under the preserved object-type rules; no applicable field is null. A closed row has every *_or_null null and fcntl(F_GETFD) must return -1/EBADF. extra_fds is exactly [].",
        "schema": "temporac.linux-fd-readback.v5",
        "stages": [
          "post-carrier-exec-entry",
          "locked-pre-cpython",
          "post-pyinitialize"
        ],
        "transitions": {
          "discovery-run-0|discovery-run-1": [
            "At post-carrier-exec-entry FD7 is closed/EBADF and fds 0,1,2,6,8 are present with descriptor flags zero; no other fd is open.",
            "Set FD_CLOEXEC on each operational fd in ascending order and require exact readback. Pread and validate fds 0 and 8 from offset zero without moving offsets; then close 0 and 8 in ascending order and prove EBADF. Retain 1,2,6.",
            "locked-pre-cpython and post-pyinitialize each contain rows for all five operational fd numbers: 1,2,6 present and locked/unchanged, 0,8 closed/null. No CPython API occurs before locked-pre-cpython is committed."
          ],
          "full-runtime": [
            "At post-carrier-exec-entry FD7 is closed/EBADF and fds 0,1,2,3,4,5,6,8 are present with descriptor flags zero; no other fd is open.",
            "Set FD_CLOEXEC on each operational fd in ascending order and require exact readback. Pread and validate fds 0,3,4,5,8 from offset zero without moving offsets; then close 3,4,5,8 in ascending order and prove EBADF. Retain 0,1,2,6.",
            "locked-pre-cpython and post-pyinitialize each contain rows for all eight operational fd numbers: 0,1,2,6 present and locked/unchanged, 3,4,5,8 closed/null. No CPython API occurs before locked-pre-cpython is committed.",
            "Dispatch consumes the already validated exact FD0 invocation bytes while FD0 remains present; immediately after the one dispatch return or exception mapping, close FD0 and prove EBADF before terminal framing. No later process or exec is permitted."
          ]
        }
      },
      "launch_evidence_index": {
        "hash_rule": "launch_evidence_index_sha256 is SHA-256 of the complete canonical index bytes including LF. The index contains neither its own digest nor any discovery output, final trace, durability attestation, result, capability or receipt digest.",
        "row_closed_keys": [
          "carrier_construction_bytes_hex",
          "carrier_construction_sha256",
          "carrier_readback_bytes_hex",
          "carrier_readback_sha256",
          "launch_instance_bytes_hex",
          "launch_instance_sha256",
          "operational_fd_readback_index_bytes_hex",
          "operational_fd_readback_index_sha256",
          "operational_memfd_construction_index_bytes_hex",
          "operational_memfd_construction_index_sha256",
          "process_ordinal",
          "schema"
        ],
        "row_rule": "Every bytes_hex value is lowercase complete canonical bytes including LF and hashes to its paired digest. Launch bytes validate the named controller request and exact mode. Operational memfd construction covers exactly the mode-specific index and excludes FD7; operational FD readback contains the three exact post-carrier stages and binds that launch. Carrier construction is the external eight-stage FD7 evidence, carrier readback is the joined child's exact v5 readback, and every comparable identity/seal/count/hash/offset field agrees under the explicit stage transition. The row contains no child output/stream/join value.",
        "row_schema": "temporac.native-launch-evidence-row.v5",
        "rows": "Mode discover-twin has exactly two rows ordered process_ordinal 0 then 1 and both bind the one top-level request digest; mode full-runtime has exactly one row with ordinal 0. Missing, extra, duplicate, reordered or cross-request row fails.",
        "schema": "temporac.native-launch-evidence-index.v5",
        "top_level_closed_keys": [
          "controller_request_sha256",
          "mode",
          "rows",
          "schema"
        ]
      },
      "launch_instance": {
        "closed_keys": [
          "argv",
          "contract_sha256",
          "controller_request_sha256",
          "cwd",
          "environment_sha256_or_null",
          "fd_rows",
          "launcher_sha256",
          "mode",
          "rlimit_nofile_cur",
          "rlimit_nofile_max",
          "runtime_ledger_bundle_sha256",
          "schema",
          "sealed_memfd_construction_index_bytes_hex",
          "sealed_memfd_construction_index_sha256",
          "transport_contract_sha256"
        ],
        "mode_tokens": [
          "discovery-run-0",
          "discovery-run-1",
          "full-runtime"
        ],
        "request_mode_mapping": "A controller request with mode=discover-twin creates exactly two launch instances in order: process_ordinal0 has mode=discovery-run-0 and process_ordinal1 has mode=discovery-run-1. A request with mode=full-runtime creates exactly one process_ordinal0 launch with mode=full-runtime. No request contains a child-mode token, no launch contains discover-twin, and no caller selects or swaps process ordinal/mode.",
        "rules": [
          "argv is always the exact child argv; mode comes only from launch bytes, never argv or environment.",
          "Discovery environment is null, fd_rows are the exact discovery operational rows, and bundle hash is BASE. Full environment is present, fd_rows are the exact full operational rows, and bundle hash is FINAL. Every mode selects the preserved pyconfig_full_runtime_final_values byte-for-byte under cpython_profile_overlay because its exact site-packages path is required by the common pre-dispatch import plan; the old stdlib-only discovery map is retained but unselected. Actual OS argv and selected PyConfig/sys readbacks are independently exact; neither is inferred from the other.",
          "sealed_memfd_construction_index_bytes_hex contains complete canonical construction-index bytes for only the listed operational memfds; its digest must match. FD7 is excluded.",
          "rlimit_nofile_cur and rlimit_nofile_max are both exact integer 256 and equal the controller pre-request lock plus every child readback.",
          "launcher_sha256 binds the exact installed supervisor ELF; controller_request_sha256 binds the exact external request; transport_contract_sha256 binds the BASE transport artifact."
        ],
        "schema": "temporac.native-launch-instance.v5"
      },
      "operational_memfd_construction": {
        "applicable_rows": {
          "discovery-run-0|discovery-run-1": [
            "0:discovery-controller-input",
            "8:base-runtime-ledger-bundle"
          ],
          "full-runtime": [
            "0:invocation",
            "3:effective-contract",
            "4:source-allowlist",
            "5:cpu-environment",
            "8:final-runtime-ledger-bundle"
          ]
        },
        "failure_mapping": "No procedure-specific error object, errno, substage token, partial construction index or diagnostic bytes are serialized or persisted. Any first syscall, short-operation, value, byte/hash, seal, identity, offset, EOF, close or cleanup failure in this ordered procedure selects only the universal three-key temporac.native-failure.v5 with stage=OPERATIONAL_MEMFD_CONSTRUCTION and launch_instance_sha256_or_null=null, closes/proves every created raw/writer/reader/target descriptor, creates no launch or child, restores controller output under the fixed rule and exits 123; unusable result output alone follows exit 124. EINTR is retried only in the explicitly named pwrite/pread chunk loops without offset advancement and is a failure for every operation_order step specified as one call.",
        "index": {
          "digest_rule": "operational_memfd_construction_index_sha256 is SHA-256 of the complete standalone canonical index bytes including LF. The digest is external and absent from the index.",
          "row_closed_keys": [
            "child_fd",
            "readback_bytes_hex",
            "readback_sha256",
            "role"
          ],
          "row_order": "Exactly the applicable rows in ascending child_fd order. readback_bytes_hex is lowercase complete temporac.operational-memfd-construction-readback.v5 bytes including LF and hashes exactly; embedded mode/child_fd/role/payload agree with launch. Missing, extra, duplicate, reordered, digest-only or cross-launch evidence fails.",
          "schema": "temporac.operational-memfd-construction-index.v5",
          "top_level_closed_keys": [
            "mode",
            "rows",
            "schema"
          ]
        },
        "memfd_name_rule": "The memfd name is exact ASCII temporac.<mode>.<role>.v5 with the exact launch mode and row role tokens, no NUL, truncation, caller suffix or alternate spelling. The procfs symlink is transport only and never an authority path.",
        "operation_order": [
          "Call memfd_create(exact name,MFD_ALLOW_SEALING|MFD_CLOEXEC) once. Under the already verified table with no descriptor at or above 64, promote the raw result with fcntl(F_DUPFD_CLOEXEC,64) and require writer_fd exactly 64; close the raw descriptor and prove F_GETFD=-1/EBADF. The sole writer is a regular anonymous memfd with FD_CLOEXEC, exact O_RDWR|O_LARGEFILE, offset zero, size zero and seals zero.",
          "Call ftruncate(writer,payload_bytes) once. Populate with an exact offset pwrite loop using chunk_bytes=1048576: for offset 0,1048576,... call pwrite(writer,payload+offset,min(1048576,payload_bytes-offset),offset), retry only EINTR without changing offset, require each non-EINTR return equals the requested chunk, and stop exactly at payload_bytes; zero/short/other error fails. Writer offset remains zero. Verify by the analogous exact pread chunk loop from offset zero with the same chunk partition while streaming byte equality and SHA-256, then one pread of one byte at payload_bytes must return zero. This loop, not one oversized syscall, applies to every payload up to ledger_bundle_max_bytes.",
          "While the sole writer remains open call F_ADD_SEALS once with exactly F_SEAL_GROW|F_SEAL_SEAL|F_SEAL_SHRINK|F_SEAL_WRITE; require return zero and exact F_GET_SEALS. No writable map or second writer exists.",
          "Open exact ASCII /proc/self/fd/64 once with O_RDONLY|O_CLOEXEC and no mode argument. Because the final component is necessarily a procfs symlink, O_NOFOLLOW is forbidden. With writer 64 still open and no other descriptor at or above 64, promote the raw reader with F_DUPFD_CLOEXEC minimum 64 and require reader_fd exactly 65; close/prove-EBADF the raw descriptor, and require exact O_RDONLY|O_LARGEFILE plus FD_CLOEXEC.",
          "With writer and reader open require identical regular-file st_dev/st_ino/size and seals, exact payload/hash/EOF, and prove distinct open-file descriptions only by lseek(reader,1,SEEK_SET)==1, lseek(writer,0,SEEK_CUR)==0, lseek(reader,0,SEEK_SET)==0, then both offsets zero.",
          "Close writer exactly once and prove F_GETFD=-1/EBADF. The reader must retain identical identity, size, seals, O_RDONLY|O_LARGEFILE, FD_CLOEXEC, offset zero and exact payload/hash/EOF.",
          "Call dup3(reader,child_fd,0) exactly once, require return child_fd, close reader and prove EBADF. The installed child fd has descriptor flags zero, O_RDONLY|O_LARGEFILE, offset zero and the identical identity/size/seals/payload/hash/EOF.",
          "Serialize the complete readback before launch construction. Transient descriptors are absent from the launch table. Reopen after writer close, dup of writer, O_NOFOLLOW on the procfs open, writable target, mmap, disk fallback, shared OFD, retry, missing/reordered step or leaked descriptor fails."
        ],
        "readback": {
          "closed_keys": [
            "child_fd",
            "mode",
            "payload_bytes",
            "payload_sha256",
            "reader_fd",
            "role",
            "rows",
            "schema",
            "writer_fd"
          ],
          "fd_value_rule": "For every applicable operational memfd row writer_fd=64 and reader_fd=65 exactly. The preconstruction descriptor scan proves both initially free; F_DUPFD_CLOEXEC is lowest-free, and no other allocation occurs between writer and reader promotions. Any other value, hidden descriptor or RLIMIT below 66 fails before launch.",
          "row_closed_keys": [
            "child_descriptor_flags_u32_or_null",
            "child_status_flags_u32_or_null",
            "reader_descriptor_flags_u32_or_null",
            "reader_offset_or_null",
            "reader_status_flags_u32_or_null",
            "same_device_or_null",
            "same_inode_or_null",
            "same_open_file_description_or_null",
            "seals",
            "stage",
            "writer_descriptor_flags_u32_or_null",
            "writer_offset_or_null",
            "writer_status_flags_u32_or_null"
          ],
          "row_order": [
            "writer-created",
            "writer-populated-verified",
            "writer-sealed",
            "reader-opened",
            "ofd-independence-proved",
            "writer-closed",
            "reader-postclose-verified",
            "child-fd-installed"
          ],
          "schema": "temporac.operational-memfd-construction-readback.v5",
          "stage_value_rules": {
            "child-fd-installed": "Child descriptor/status are 0 and O_RDONLY|O_LARGEFILE; reader/writer fields null; seals exact four-token list; same_device=true,same_inode=true,same_open_file_description=true compare child with the immediately preceding reader snapshot.",
            "ofd-independence-proved": "Writer and reader descriptor/status/offset are FD_CLOEXEC,O_RDWR|O_LARGEFILE,0 and FD_CLOEXEC,O_RDONLY|O_LARGEFILE,0 respectively; child fields null; seals exact; all same_* are true,true,false.",
            "reader-opened": "Writer and reader descriptor/status/offset have the same exact values as the next proof stage; child fields null; seals exact; every same_* is null pending the ordered proof.",
            "reader-postclose-verified": "Reader descriptor/status/offset are FD_CLOEXEC,O_RDONLY|O_LARGEFILE,0; writer/child fields null; seals exact; all same_* retain true,true,false.",
            "writer-closed": "Reader values remain FD_CLOEXEC,O_RDONLY|O_LARGEFILE,0; writer/child fields are null after EBADF; seals exact; all same_* retain true,true,false.",
            "writer-created": "Writer descriptor/status/offset are FD_CLOEXEC,O_RDWR|O_LARGEFILE,0; reader/child and all same_* fields null; seals [].",
            "writer-populated-verified": "Exactly writer-created field values and seals []; the top-level payload bytes/hash, exact size and EOF have been verified.",
            "writer-sealed": "Writer descriptor/status/offset remain FD_CLOEXEC,O_RDWR|O_LARGEFILE,0; reader/child and same_* fields null; seals exactly [F_SEAL_GROW,F_SEAL_SEAL,F_SEAL_SHRINK,F_SEAL_WRITE]."
          }
        },
        "schema": "temporac.operational-memfd-construction-procedure.v5"
      },
      "transport_contract": {
        "closed_keys": [
          "contract_sha256",
          "rows",
          "schema"
        ],
        "digest_rule": "transport_contract_sha256 is SHA-256 of the complete canonical artifact bytes including LF. The digest is external and absent from the object; the object contains no launch bytes/hash or run-specific evidence.",
        "row_closed_keys": [
          "json_pointer",
          "value_sha256"
        ],
        "row_order": [
          "/round7_native_wave2_closure/controller/argv",
          "/round7_native_wave2_closure/controller/child_fd_actions",
          "/round7_native_wave2_closure/controller/durability",
          "/round7_native_wave2_closure/controller/environment",
          "/round7_native_wave2_closure/controller/external_fd_contract",
          "/round7_native_wave2_closure/controller/fork_exec",
          "/round7_native_wave2_closure/controller/modes",
          "/round7_native_wave2_closure/controller/owner",
          "/round7_native_wave2_closure/controller/request",
          "/round7_native_wave2_closure/controller/resource_limits",
          "/round7_native_wave2_closure/controller/result_stream",
          "/round7_native_wave2_closure/controller/side_effect_boundary",
          "/round7_native_wave2_closure/framing_and_failures",
          "/round7_native_wave2_closure/generated_source_closure/controller_input",
          "/round7_native_wave2_closure/transport_and_launch/carrier",
          "/round7_native_wave2_closure/transport_and_launch/fd_layouts",
          "/round7_native_wave2_closure/transport_and_launch/fd_readback",
          "/round7_native_wave2_closure/transport_and_launch/launch_evidence_index",
          "/round7_native_wave2_closure/transport_and_launch/launch_instance",
          "/round7_native_wave2_closure/transport_and_launch/operational_memfd_construction"
        ],
        "row_rule": "Resolve each pointer in the exact accepted timestamped Amendment005 JSON and hash the complete canonical standalone value plus LF. Rows occur exactly in row_order. contract_sha256 is the accepted Round7 successor digest. The controller rows exhaust every controller key except runtime_launcher_artifact_schema; the future runtime_launcher instance is constructed later inside the final environment under that schema and binds this transport-contract digest, so hashing the schema or instance here would be a forbidden backedge. This BASE artifact fixes FD6/7/8, controller/request/child modes, carrier/memfd/readback/launch schemas, frames, caps, error/EOF/exit rules and discovery input without embedding itself.",
        "schema": "temporac.native-transport-contract.v5"
      }
    },
    "version_layers": {
      "artifact": {
        "closed_keys": [
          "contract_sha256",
          "historical_science",
          "native_provenance_verifier",
          "schema"
        ],
        "digest_rule": "version_layer_index_sha256 is SHA-256 of the complete canonical four-key artifact bytes including LF. The digest is external and absent from the artifact.",
        "historical_closed_keys": [
          "cpython",
          "operator_runtime_lock_sha256",
          "rule"
        ],
        "native_closed_keys": [
          "cpython",
          "rule"
        ],
        "rule": "contract_sha256 is the accepted Round7 successor. historical_science and native_provenance_verifier are byte-for-byte the two value objects below with exactly their respective closed keys, and schema is temporac.runtime-version-layer-index.v5. Neither row contains the artifact digest or changes a historical foreign key.",
        "schema": "temporac.runtime-version-layer-index.v5"
      },
      "historical_science": {
        "cpython": "3.12.4",
        "operator_runtime_lock_sha256": "47c2227b74b627a03bf9b5e4930dd02484caaae5a695ae5c0129f04781115cd8",
        "rule": "Existing X0/operator/science products retain their historical accepted bytes, source-snapshot foreign keys and 3.12.4 runtime claims. Live-source or Round7 supervisor bytes are never substituted into those historical records."
      },
      "native_provenance_verifier": {
        "cpython": "3.12.13",
        "rule": "The stock unpatched supervisor runtime creates provenance/discovery/bytewise replay evidence only. It may validate and recompute against exact 3.12.4-produced artifacts, but cannot relabel or replace them; any byte difference fails."
      },
      "schema": "temporac.runtime-version-layer-index.v5"
    },
    "wave2_status": "SPECIFIED_NOT_IMPLEMENTED_NOT_RUN"
  },
  "scope_lock": {
    "dominant_method_count": 1,
    "forbidden_expansion": [
      "no new algorithm, trainable component, dataset, identity, arm, seed, condition, job, GPU hour, threshold, result, claim, or evaluator access",
      "no migration of natural development data into checkpoint selection",
      "no authority follows from this document or its hashes"
    ],
    "g5a_receipt_key_count": 14,
    "hard_a6000_hour_ceiling": 100,
    "natural_development_identity_count": 134,
    "planned_a6000_hours": 93,
    "prediction_envelope_count": 9648,
    "prediction_identity_count": 402,
    "response_checkpoint_count": 480,
    "response_jobs": 24,
    "target_npz_member_count": 7,
    "target_receipt_key_count": 10,
    "teacher_checkpoint_count": 120,
    "teacher_jobs": 3,
    "training_job_count": 27,
    "unchanged": [
      "problem anchor and claims",
      "data and split boundary",
      "27 exact job names and three seeds",
      "fixed optimizer steps and checkpoint cadence",
      "93 planned A6000-hours and 100-hour ceiling",
      "all scientific thresholds and kill semantics",
      "four prediction arms and two conditions",
      "existing seven-member target artifact",
      "existing ten-key temporac.target-receipt.v4",
      "existing fourteen-key temporac.g5a-receipt.v4",
      "one-use capability and non-resumable evaluator"
    ]
  },
  "score_and_selection": {
    "g1_receipt": {
      "closed_keys": [
        "code_index_sha256",
        "contract_sha256",
        "environment_sha256",
        "schema",
        "score_index_sha256",
        "selection_receipt_sha256",
        "status",
        "teacher_checkpoint_evidence_index_sha256",
        "teacher_run_receipt_sha256",
        "teacher_tune_input_receipt_sha256"
      ],
      "exact_key_count": 10,
      "field_types": {
        "teacher_run_receipt_sha256": "exact JSON array of exactly three lowercase SHA-256 strings in seed order [20260815,20260816,20260817]; never a scalar or aggregate digest"
      },
      "schema": "temporac.g1-teacher-selection-receipt.v4",
      "status": "PASS only after the exact length-three teacher-run array, shared tune input, complete 120-row checkpoint evidence, all artifacts/evaluations/scores, and selection independently replay"
    },
    "global_k7_replay": [
      "After G5b PASS and before any per-identity K7 replay, perform this selection replay exactly once for the evaluator process.",
      "Rebuild and byte-verify the shared tune input; validate the exact three-element teacher-run receipt array in seed order 20260815/20260816/20260817; load and member-replay all 120 checkpoint artifacts; rerun all 120x56 forwards; require every tune output NPZ and evaluation receipt byte-identical; require every score-row binary64 bit; rerun per-seed and global precedence; and require selection and G1 receipt bytes.",
      "Only this replay constructs one typed SelectedTeacher. All 120 checkpoint artifacts/receipts, 120 tune output artifacts/evaluation receipts, the shared tune input artifact/receipt, score index, selection receipt, evidence index, G1 receipt, and three run receipts are required. A missing or tampered loser fails before per-identity replay."
    ],
    "global_selection": "From the three per-seed winners choose lexicographic minimum (tune_abstentions,mean_masked_reconstruction,tune_objective,seed), exactly as the proposal.",
    "per_seed_selection": "For each seed discard a row when negative_edge_fraction>0.01 or any required value/evidence is invalid; choose lexicographic minimum (tune_objective,step). Missing all candidates for a seed fails G1.",
    "score_index": {
      "complete_grid": "seeds 20260815, 20260816, 20260817 crossed with steps 500,1000,...,20000 exactly once",
      "order": "seed numeric ascending, then completed step numeric ascending",
      "row_closed_keys": [
        "checkpoint_receipt_sha256",
        "mean_masked_reconstruction",
        "negative_edge_fraction",
        "seed",
        "step",
        "tune_abstentions",
        "tune_evaluation_receipt_sha256",
        "tune_objective"
      ],
      "row_count": 120,
      "row_exact_key_count": 8,
      "schema": "temporac.teacher-score-index.v4",
      "score_rule": "Each score field mirrors the named tune-evaluation receipt. Parse JSON numbers to binary64 and require exact IEEE-754 bits equal independent recomputation; -0.0, NaN, infinity, string numbers, rounding tolerance, or caller rewrite fails.",
      "top_level_closed_keys": [
        "rows",
        "schema"
      ]
    },
    "selected_teacher_type": "SelectedTeacher is an opaque package-private capability containing the freshly strict-loaded eval-mode CPU module plus immutable checkpoint artifact/receipt, selection receipt, score-index, code-index, effective-contract, and environment digests. Its constructor is private to the successful one-global selection replay and it cannot be deserialized from a digest.",
    "selection_receipt": {
      "closed_keys": [
        "code_index_sha256",
        "contract_sha256",
        "environment_sha256",
        "per_seed_winner_checkpoint_receipt_sha256",
        "schema",
        "score_index_bytes",
        "score_index_sha256",
        "selected_checkpoint_artifact_sha256",
        "selected_checkpoint_receipt_sha256",
        "teacher_tune_input_artifact_sha256",
        "teacher_tune_input_receipt_sha256"
      ],
      "exact_key_count": 11,
      "per_seed_winner_order": "three checkpoint-receipt digests in seed order 20260815, 20260816, 20260817",
      "schema": "temporac.teacher-selection-receipt.v4"
    },
    "teacher_checkpoint_evidence_index": {
      "field_types": {
        "teacher_run_receipt_sha256": "exact JSON array of exactly three lowercase SHA-256 strings in seed order [20260815,20260816,20260817]; every one of 120 rows maps to the element matching its seed"
      },
      "order": "seed then step",
      "retention": "Every named checkpoint artifact/receipt and tune output/evaluation receipt remains immutable and readable through K7.",
      "row_closed_keys": [
        "checkpoint_artifact_sha256",
        "checkpoint_receipt_sha256",
        "job_name_hex",
        "seed",
        "step",
        "tune_output_artifact_sha256",
        "tune_evaluation_receipt_sha256"
      ],
      "row_count": 120,
      "schema": "temporac.teacher-checkpoint-evidence-index.v4",
      "top_level_closed_keys": [
        "contract_sha256",
        "rows",
        "schema",
        "teacher_run_receipt_sha256",
        "teacher_tune_input_receipt_sha256"
      ]
    }
  },
  "status": "PROPOSED_PENDING_ROUND7_REVIEW",
  "teacher_checkpoint": {
    "artifact": "deterministic uncompressed sorted NPZ with exactly 16 state_dict members; every array NPY v2.0 C-order <f4; no pickle or extra member",
    "artifact_count": 120,
    "checkpoint_steps": [
      500,
      1000,
      1500,
      2000,
      2500,
      3000,
      3500,
      4000,
      4500,
      5000,
      5500,
      6000,
      6500,
      7000,
      7500,
      8000,
      8500,
      9000,
      9500,
      10000,
      10500,
      11000,
      11500,
      12000,
      12500,
      13000,
      13500,
      14000,
      14500,
      15000,
      15500,
      16000,
      16500,
      17000,
      17500,
      18000,
      18500,
      19000,
      19500,
      20000
    ],
    "loader": [
      "Read exact artifact and receipt bytes, recompute both hashes, validate the closed receipt and exact code/config/effective-contract/environment/run/job/seed/step bindings.",
      "Parse with allow_pickle=false and the checkpoint-specific 2 MiB archive cap; recompute every NPY member byte ledger; require exact sorted names, shapes, <f4 dtype, C order, finiteness, and no extra key.",
      "Instantiate a fresh CPU TempoRACTeacher in the bound environment, copy no caller module or state object, strict-load all 16 members, reserialize the loaded state with the same writer, and require byte-for-byte equality with the input artifact.",
      "Set eval mode; gradients disabled. teacher_sha256 is only SHA-256 of these complete exact checkpoint bytes.",
      "A digest, caller state_dict, untyped module, cached phase, or cached reconstruction cannot construct SelectedTeacher."
    ],
    "members": [
      {
        "dtype": "<f4",
        "name": "phase.layers.0.bias",
        "shape": [
          256
        ]
      },
      {
        "dtype": "<f4",
        "name": "phase.layers.0.weight",
        "shape": [
          256,
          247
        ]
      },
      {
        "dtype": "<f4",
        "name": "phase.layers.2.bias",
        "shape": [
          128
        ]
      },
      {
        "dtype": "<f4",
        "name": "phase.layers.2.weight",
        "shape": [
          128,
          256
        ]
      },
      {
        "dtype": "<f4",
        "name": "phase.layers.4.bias",
        "shape": [
          2
        ]
      },
      {
        "dtype": "<f4",
        "name": "phase.layers.4.weight",
        "shape": [
          2,
          128
        ]
      },
      {
        "dtype": "<f4",
        "name": "reconstruction.layers.0.bias",
        "shape": [
          256
        ]
      },
      {
        "dtype": "<f4",
        "name": "reconstruction.layers.0.weight",
        "shape": [
          256,
          34
        ]
      },
      {
        "dtype": "<f4",
        "name": "reconstruction.layers.2.bias",
        "shape": [
          256
        ]
      },
      {
        "dtype": "<f4",
        "name": "reconstruction.layers.2.weight",
        "shape": [
          256,
          256
        ]
      },
      {
        "dtype": "<f4",
        "name": "reconstruction.layers.4.bias",
        "shape": [
          149
        ]
      },
      {
        "dtype": "<f4",
        "name": "reconstruction.layers.4.weight",
        "shape": [
          149,
          256
        ]
      },
      {
        "dtype": "<f4",
        "name": "static.layers.0.bias",
        "shape": [
          128
        ]
      },
      {
        "dtype": "<f4",
        "name": "static.layers.0.weight",
        "shape": [
          128,
          298
        ]
      },
      {
        "dtype": "<f4",
        "name": "static.layers.2.bias",
        "shape": [
          32
        ]
      },
      {
        "dtype": "<f4",
        "name": "static.layers.2.weight",
        "shape": [
          32,
          128
        ]
      }
    ],
    "raw_payload_bytes": 1008348,
    "receipt": {
      "closed_keys": [
        "artifact_bytes",
        "artifact_sha256",
        "code_index_sha256",
        "config_sha256",
        "contract_sha256",
        "environment_sha256",
        "job_name_hex",
        "members",
        "run_receipt_sha256",
        "schema",
        "seed",
        "step"
      ],
      "exact_key_count": 12,
      "member_closed_keys": [
        "bytes",
        "dtype",
        "name",
        "sha256",
        "shape"
      ],
      "schema": "temporac.teacher-checkpoint-receipt.v4"
    },
    "schema": "temporac.teacher-checkpoint.v4",
    "schema_specific_archive_cap_bytes": 2097152,
    "seeds": [
      20260815,
      20260816,
      20260817
    ]
  },
  "teacher_tune_input": {
    "construction": [
      "Rebuild all 56 canonical X0 tune views from the exact 40-row X0 manifest and effective code/config/environment; bytewise equality with the shared artifact is mandatory at G1 and again in the one global K7 selection replay.",
      "sample_offsets[0]=0, sample_offsets[56]=25680, entries strictly increase, and view v owns samples [sample_offsets[v],sample_offsets[v+1]).",
      "edge_offsets[v]=sample_offsets[v]-v; view v owns analytic edge rows [edge_offsets[v],edge_offsets[v+1]). This derives all 56 per-view E_v=N_v-1 partitions without another member.",
      "For each view the canonical adapter creates exactly one CertificateTrack run with C-order <i4 run_bounds [[0,N_v]]. integer_landmarks on that run must rederive exactly ten ordered traversal intervals; traversal_bounds[v] stores those intervals relative to the view slice.",
      "Each traversal bound denotes the half-open owned edge interval [left,right), with samples left through right. The stored traversal_bounds member is only the analytic/F11 ledger and is never passed as run_bounds.",
      "teacher_input is the canonical C-order float32 215-channel F4 input. static_features is the one-time C-order float32 cast of the normative F5 binary64 298-vector for the same view.",
      "analytic_chi and analytic_pulse are used only for X0 certify_target analytic equality checks; they never substitute for checkpoint phase or reconstruction."
    ],
    "edge_count": 25624,
    "forbidden_inputs": "natural rows, evaluator labels, count, period, density, boundary, bbox, supplied identity names, or any teacher output",
    "members": [
      {
        "dtype": "<f8",
        "name": "analytic_chi",
        "shape": [
          25624
        ]
      },
      {
        "dtype": "|u1",
        "name": "analytic_pulse",
        "shape": [
          25624
        ]
      },
      {
        "dtype": "<i8",
        "name": "sample_offsets",
        "shape": [
          57
        ]
      },
      {
        "dtype": "<i8",
        "name": "sampled_source_clock",
        "shape": [
          25680
        ]
      },
      {
        "dtype": "<f4",
        "name": "static_features",
        "shape": [
          56,
          298
        ]
      },
      {
        "dtype": "<f4",
        "name": "teacher_input",
        "shape": [
          25680,
          215
        ]
      },
      {
        "dtype": "<i4",
        "name": "traversal_bounds",
        "shape": [
          56,
          10,
          2
        ]
      }
    ],
    "raw_payload_bytes": 22592544,
    "receipt": {
      "closed_keys": [
        "artifact_bytes",
        "artifact_sha256",
        "code_index_sha256",
        "config_sha256",
        "contract_sha256",
        "environment_sha256",
        "members",
        "schema",
        "x0_manifest_sha256"
      ],
      "exact_key_count": 9,
      "member_closed_keys": [
        "bytes",
        "dtype",
        "name",
        "sha256",
        "shape"
      ],
      "schema": "temporac.teacher-tune-input-receipt.v4",
      "validation": "artifact byte count/hash, exact seven member ledger, view roster, X0 manifest, code/config/effective-contract/environment, raw payload, archive cap, semantic partitions, and deterministic reserialization all match"
    },
    "sample_count": 25680,
    "schema": "temporac.teacher-tune-input.v4",
    "schema_specific_archive_cap_bytes": 33554432,
    "view_count": 56,
    "view_order": "source_id 24 through 31 outer, block_index 0 through 6 inner; linear resampler and direct offset 0 only; view v=(source_id-24)*7+block_index"
  },
  "tune_output_and_scoring": {
    "evaluation_execution": [
      "For every checkpoint, strict-load it and independently rebuild and byte-verify the shared tune-input artifact.",
      "Visit views v=0 through 55. For each view let N_v=sample_offsets[v+1]-sample_offsets[v], run geometry-only integer_landmarks on exactly one run, and construct CertificateTrack.run_bounds as the exact C-order array np.asarray([[0,N_v]],dtype='<i4'). Issue exactly one CPU float32 forward using teacher_input[sample_offsets[v]:sample_offsets[v+1]] and static_features[v].",
      "The ten stored traversal_bounds[v] rows must equal the ten geometry/integer-landmark traversal bounds rederived from that one run. They are used only by analytic checks and the F11 score ledger and are never supplied as CertificateTrack.run_bounds.",
      "Use eval plus inference_mode, one intra-op thread, one inter-op thread, no autocast/TF32, and the exact locked and probed floating-point environment.",
      "Concatenate output rows only in view/sample order into the exact two-member output NPZ. Require shapes, C order, finiteness, deterministic reserialization, raw payload, and 16 MiB cap.",
      "The full typed X0 track for each view is reconstructed from tune input plus that checkpoint output. Every tune abstention invokes complete certify_target with the one-run run_bounds; analytic certificate_track shortcut is forbidden."
    ],
    "evaluation_receipt": {
      "closed_keys": [
        "artifact_bytes",
        "artifact_sha256",
        "checkpoint_receipt_sha256",
        "code_index_sha256",
        "config_sha256",
        "contract_sha256",
        "environment_sha256",
        "members",
        "schema",
        "scores",
        "seed",
        "step",
        "teacher_tune_input_receipt_sha256"
      ],
      "exact_key_count": 13,
      "independence": "This receipt is built from the recomputed output artifact and score arithmetic, not copied from the score index. Its hash is the authority-bearing provenance named by one score row.",
      "member_closed_keys": [
        "bytes",
        "dtype",
        "name",
        "sha256",
        "shape"
      ],
      "schema": "temporac.teacher-tune-evaluation-receipt.v4",
      "scores_closed_keys": [
        "mean_masked_reconstruction",
        "negative_edge_fraction",
        "tune_abstentions",
        "tune_objective"
      ]
    },
    "exact_reductions": {
      "binary64_normalized_smooth_l1": {
        "boundary": "Use the quadratic branch iff e <= delta; equality is quadratic. delta is the binary64 value obtained by parsing JSON number 0.05, bits 0x3fa999999999999a.",
        "casts_and_inputs": "For each masked element, promote the exact stored <f4 reconstruction value to binary64 once and promote the exact corresponding stored <f4 teacher_input target value to binary64 once. Let p and y denote those exact promoted binary64 values; there is no decimal reparse or earlier binary64 surrogate.",
        "evaluation_order": [
          "d = fl64(p - y)",
          "e = abs(d)",
          "if e <= delta: q = fl64(e * e); h = fl64(0.5 * q); loss = fl64(h / delta)",
          "if e > delta: h = fl64(0.5 * delta); loss = fl64(e - h)"
        ],
        "formula": "This is normalized SmoothL1: 0.5*e^2/delta for e<=delta, otherwise e-0.5*delta. The unnormalized Huber alternative 0.5*e^2 for e<=delta and delta*(e-0.5*delta) otherwise is forbidden.",
        "operation_rules": "Every named operation rounds once to IEEE-754 binary64 under FE_TONEAREST; no FMA, reassociation, approximate reciprocal, extended-precision retention, fused/vector reduction, or algebraically equivalent rewrite is permitted. Mask predicates are evaluated before loss and masked-out elements perform no score arithmetic."
      },
      "correspondence": [
        "For each nonreference block 1 through 6, evaluate the 128 fractions (j+1/2)/128 in j order against block 0 using F9 inversion, right-tie ownership, shortest signed principal-arc interpolation, and term 1-dot.",
        "Neumaier equal-average fraction, traversal, nonreference block, then source_id in that order."
      ],
      "domains": "tune_abstentions is an integer in [0,56]; negative_edge_fraction is finite in [0,1]; mean_masked_reconstruction and tune_objective are finite, nonnegative binary64; every computed zero is canonical +0.0.",
      "negative_edge_fraction": "Count, with integer arithmetic, every traversal-owned valid edge whose increment is strictly less than -1e-12 and divide once in binary64 by the positive integer count of all traversal-owned valid edges. Do not average ratios by view.",
      "neumaier": "For ordered binary64 terms x: t=s+x; compensation adds (s-t)+x when abs(s)>=abs(x), otherwise (x-t)+s; final value s+compensation. Empty or nonpositive denominator fails.",
      "phase_penalties": [
        "A traversal owns edges e=left through right-1 exactly once. Valid means the existing F11 observed-retained-edge predicate holds and both normalized phase endpoints are finite.",
        "For each owned valid edge in increasing e, compute the F11 signed principal increment in cycles. Neumaier-average edge, traversal, block, then source.",
        "orientation is mean ReLU(-increment); alias is mean ReLU(abs(increment)-0.25)."
      ],
      "reconstruction": [
        "Use the exact binary64 normalized SmoothL1 record with delta 0.05 between exact promoted <f4 checkpoint reconstruction and exact promoted first-149-channel <f4 teacher_input target, under masks derived from the last 66 teacher-input mask channels.",
        "Within each traversal, sample indices increase; at each sample channels 0 through 148 are the innermost order. Neumaier-sum only masked channel losses and convert the positive integer mask count to binary64 exactly once before one final division.",
        "Neumaier equal-average traversal 0 through 9, then block 0 through 6, then source_id 24 through 31. Each integer group count is converted to binary64 once immediately before its division. This value is mean_masked_reconstruction and exactly F11 L_rec on the tune inventory."
      ],
      "tune_abstentions": "Count integer ABSTAIN results from full certify_target over all 56 views. Partial checks, phase-only checks, analytic phase/reconstruction, or certificate_track are not tune outputs.",
      "tune_objective": "Using the exact already-rounded binary64 scalars, compute a=fl64(0.25*orientation), b=fl64(0.25*alias), c=fl64(L_rec+L_corr), d=fl64(c+a), tune_objective=fl64(d+b). No reassociation or FMA."
    },
    "output_count": 120,
    "output_members": [
      {
        "dtype": "<f4",
        "name": "phase",
        "shape": [
          25680,
          2
        ]
      },
      {
        "dtype": "<f4",
        "name": "reconstruction",
        "shape": [
          25680,
          149
        ]
      }
    ],
    "output_raw_payload_bytes": 15510720,
    "output_schema": "temporac.teacher-tune-output.v4",
    "output_schema_specific_archive_cap_bytes": 16777216,
    "typed_view_reconstruction": "For view v, continuous is teacher_input slice columns 0:149 cast once to C-order <f8; continuous_mask is geometry element mask from columns 149:182 repeated over x/y for channels 0:66, direction element mask from 182:215 repeated over x/y for 66:132, and geometry element-mask entries 0:17 for confidence 132:149. Samples/clocks use sample_offsets; analytic edges and F11 grouping use traversal_bounds row v. CertificateTrack.run_bounds is separately exact <i4[[0,N_v]] and never traversal_bounds[v]. No caller field participates."
  }
}
```
