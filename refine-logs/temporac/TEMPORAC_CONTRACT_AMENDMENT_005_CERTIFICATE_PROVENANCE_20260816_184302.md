# TempoRAC Contract Amendment 005 - Certificate Provenance and Closed K7 Replay (Round 5)

Status: PROPOSED_PENDING_ROUND5_REVIEW  
Family: same-family provisional  
Authority: all zero  
Generated: 2026-08-16T17:35:27+08:00

This document is the complete Round-5 replacement for the rejected Round-4 Amendment005 candidate. It is a provenance specification, not an implementation, accepted review, gate receipt, test run, experiment, or authorization. It closes only fresh-review findings A005-R4-B1, A005-R4-B2, and A005-R4-B3 while preserving the already closed R3-B2 graph and every scientific/interface rule. It does not alter the dominant method, algorithm, data, claims, thresholds, exact 27 training jobs, 93 planned A6000-hours, 402-identity population, 9,648 prediction envelopes, seven-member target artifact, ten-key target receipt, fourteen-key G5a receipt, capability semantics, or evaluator metrics.

## 1. Authority, precedence, and immutable scope

Every authority bit is zero. This amendment grants no P0/P00, P1/P01, P2, P2-METRIC/P05M, P3/P06, S0, gate, data, server, GPU, training, test, vault, capability, evaluator, result, Git, or paper-claim authority. No future artifact, receipt, environment, launch instance, discovery output, trace, or test described here exists merely because its schema is specified.

Only a fresh independent Round-5 ACCEPT of the exact timestamped MD/JSON pair may permit a later separately authorized implementation and review. If accepted, this amendment refines the canonical proposal only where it speaks expressly. All unmentioned canonical rules remain unchanged.

The scientific and resource boundary remains exact:

- one dominant TempoRAC method;
- three teacher jobs and 24 response jobs, exactly 27 training jobs;
- seeds 20260815, 20260816, and 20260817;
- 120 teacher checkpoints and 480 response checkpoints;
- 93 planned A6000-hours under the existing 100-hour ceiling;
- 268 train plus 134 development identities, exactly P402;
- four arms, three seeds, two conditions, exactly 9,648 prediction envelopes;
- the existing seven-member target NPZ and ten-key temporac.target-receipt.v4;
- the existing fourteen-key temporac.g5a-receipt.v4;
- unchanged thresholds, kill semantics, capability consumption, evaluator population, and claims.

There is no new job, model, loss weight, trainable parameter, seed, identity, dataset row, prediction cell, target member, threshold, result, or claim.

## 2. Bound snapshot and byte-identical history

The canonical proposal SHA-256 remains 3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391. The current plan and tracker hashes remain 4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e and 714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b. The certificate postfix Round2 MD/JSON hashes remain 93a9cae771b34c185b9057286794b894d615aec5424011df145aaf8900f008d8 and 144d2f22f4f75166b8cbefe54157d08690408979dd7cb34ef8c97e1738432872.

The fresh Round-4 review that requires this revision is bound exactly:

- MD: 18,713 bytes, 988eb7c2b61eb445a07e1c17b7bcf44be256f9948d6600a6ed6edc62ef71cfc1;
- JSON: 12,977 bytes, f8552023abb52c98baec6a9143d15e93e3268d883bf3967c446c3fcbcd9d06d9.

The rejected Round-4 fixed pair is preserved byte-identically under new archive timestamp 20260816_173526:

- MD: 203,317 bytes, 0552aaca74b34929ead1124cce846d2a583931aa89ad29fcf0f97486b7be0198;
- JSON: 165,291 bytes, 04d24ad19b19470bcd675974766a279dc7691d3fdaecf665e3fd24e4a4aaeb82.

The original Round-4 timestamped pair at 20260816_154812 has those same hashes and remains unchanged. Earlier rejected archives, accepted-input reviews, and reviewer traces remain bound in the JSON lineage and are not effective-contract constituents.

The complete current local non-self snapshot has 57 paths: three runtime-bootstrap files, 24 TempoRAC source files, 19 TempoRAC tests, and eleven planning/review files. The two newly bound paths are the exact Round-4 review pair above. A generation-time full rehash matched all 57 declared digests, including tests/temporac/test_gates_fixtures.py d6d91a7dd606c42f672bb015d77c5c10406c2c6de69ed3a8bff76a0ce4f713d3 and tests/temporac/test_operator_manifest.py 7f2c90c320d7378a74fcc64f6c808eb6e426cdcfbd73cab0fb1242f690db9db0.

The CPython authority profile additionally binds exact upstream v3.12.13 bytes:

| path | bytes | SHA-256 |
|---|---:|---|
| Doc/c-api/init_config.rst | 54,271 | b6064567d191ade7c873b229d8ded27b917710c0da367343edb13488d2c63d88 |
| Include/cpython/initconfig.h | 7,820 | 86e3b9d1de6f310415912e2cdfdc276e311c026ec7fdf6190893f6313cd860a3 |
| Include/internal/pycore_initconfig.h | 5,706 | caf13e2c290ae8375636d0e1f3b1851a90396b3747da650d058c282b8743b558 |
| Lib/dataclasses.py | 62,085 | d242aea5fcf6408b1c1f622442f88f68b9526ce1f8bd2890d74a144677c427d9 |
| Python/initconfig.c | 92,299 | a50ff760d49e9d1da8a5a998bcedc8c377f12628d7d0426fc4dab89c1abfeff7 |
| Python/preconfig.c | 25,540 | 837e6e39347c4ded7d053c7f23d3ac751614a8dc135332748af7a4b0fcd46cab |

These are normative source bindings, not claims that a runtime artifact already exists.

## 3. Canonical bytes and acyclic effective contract

Canonical downstream JSON is UTF-8 without BOM, recursively sorted keys, compact comma/colon separators, ensure_ascii=false, allow_nan=false, no duplicate key, and exactly one terminal LF. Receipt hashes include that LF. Artifact hashes cover complete exact file bytes. Raw array hashes cover exact C-order element bytes only. NPY member hashes cover complete NPY-v2.0 member bytes.

H(tag,fields) is SHA256 over tag ASCII, one zero byte, then for each field uint64_be(byte_length) followed by exact field bytes. Digests are lowercase 64-hex. Integer ranges are checked before encoding. Deterministic NPZ and all existing artifact rules are unchanged.

The future temporac.effective-contract-index.v4 has exactly rows and schema. Its five rows are ordered:

1. canonical proposal, exact FINAL_PROPOSAL.md;
2. this exact timestamped Round-5 MD at suffix 20260816_173527;
3. this exact timestamped Round-5 JSON at suffix 20260816_173527;
4. future exact accepted Round-5 review MD;
5. future exact accepted Round-5 review JSON.

Each row has exactly bytes, role, sha256. The index has no contract_sha256 field. None of its five constituents contains the resulting index digest. contract_sha256 is SHA-256 of complete index bytes including LF and is assigned only after acceptance. Fixed aliases, rejected archives, prior reviews, and traces are not constituents. There is no self-hash, placeholder, guessed digest, or fixpoint.

The executable construction is forward-only:

accepted effective contract -> pre-environment supervisor/CPython/initial-ELF/init/FD ledgers -> exact discovery capsule -> durable zero-process precommit -> two bootstrap discoveries -> byte-identical output/root -> trusted-generated index -> final source/wheel/installed/environment indexes -> distinct full supervisor execution -> typed post-run trace evidence.

The capsule contains no final source-code index, installed project, trusted-generated index, final environment, trace, G5a, receipt, experiment artifact, or data digest. Trace and launch/readback evidence can reject nonmembership but cannot create, expand, or alter executable authority.

## 4. A005-R2-B1 - exact tune arithmetic and certificate input

### 4.1 Exact binary64 normalized SmoothL1

For every masked reconstruction element, promote the exact stored checkpoint reconstruction <f4 value once to binary64 and promote the exact corresponding first-149-channel teacher_input <f4 value once to binary64. Let those exact values be p and y. delta is the binary64 value of JSON number 0.05, bits 0x3fa999999999999a.

The sole loss evaluation is:

1. d = fl64(p - y);
2. e = abs(d);
3. if e <= delta, q = fl64(e * e), h = fl64(0.5 * q), loss = fl64(h / delta);
4. otherwise, h = fl64(0.5 * delta), loss = fl64(e - h).

Equality e==delta uses the quadratic branch. This is normalized SmoothL1, 0.5*e^2/delta below and at delta, e-0.5*delta above it. The unnormalized Huber alternative is forbidden. Every named operation rounds once under FE_TONEAREST. FMA, reassociation, approximate reciprocal, extended-precision retention, vector/fused reduction, or an algebraically equivalent rewrite is nonconforming.

Within one traversal, samples increase and channels 0 through 148 are innermost. Only masked terms are Neumaier-summed. The positive integer denominator is converted to binary64 exactly once before one division. Equal averaging then proceeds traversal 0..9, block 0..6, source 24..31, each with exact Neumaier order and one denominator conversion.

The complete F11 objective uses already-rounded scalars: a=fl64(0.25*orientation), b=fl64(0.25*alias), c=fl64(L_rec+L_corr), d=fl64(c+a), objective=fl64(d+b). No other parenthesization is accepted. Negative-edge fraction remains the exact integer ratio over all traversal-owned valid edges. tune_abstentions remains the count of complete certify_target ABSTAIN outcomes across all 56 views.

### 4.2 One run for certify_target; ten traversals only for ledgers

For each tune view v, N_v=sample_offsets[v+1]-sample_offsets[v]. The typed CertificateTrack has exactly one C-order run_bounds array:

np.asarray([[0,N_v]], dtype="<i4")

Geometry-only integer_landmarks runs on that one run. The exact ten traversal_bounds[v] rows must be independently rederived from it. Those ten rows partition analytic edges and define F11 traversal aggregation only. They are never passed as CertificateTrack.run_bounds. Substituting the ten rows for one run changes certification semantics and fails.

The complete certify_target path consumes actual checkpoint phase/reconstruction and the exact analytic fields. The analytic certificate_track shortcut remains forbidden.

### 4.3 Three-run G1 containers

The existing teacher_run_receipt_sha256 field in both temporac.g1-teacher-selection-receipt.v4 and temporac.teacher-checkpoint-evidence-index.v4 is an exact JSON array, not a scalar and not an aggregate digest. It contains exactly three lowercase digests ordered seeds 20260815, 20260816, 20260817. Every evidence row maps to the element for its seed. Missing, extra, duplicate, reordered, or scalar encoding fails.

## 5. A005-R2-B2 - active CPU floating-point control

The future CPU environment artifact adds a closed floating_point_control record. Future P06/P3 must measure and bind actual Python, Torch, NumPy, SciPy, wheel, ELF, BLAS, oneDNN, CPU, ISA, dispatch, helper, and thread values. This amendment fabricates none.

Before every authority-bearing numerical replay:

- call fesetround(FE_TONEAREST), require success, and read FE_TONEAREST back;
- on x86_64, use an allowlisted environment-indexed helper to write and read exact MXCSR 0x00001f80;
- call torch.set_flush_denormal(False), require supported/successful state;
- set/read torch deterministic algorithms true with warn_only false and matmul precision highest;
- set/read MKLDNN disabled and deterministic, and NNPACK disabled;
- set intra-op and inter-op threads to one and retain the existing exact BLAS/OpenMP environment;
- retain eval, inference_mode, CPU float32, no autocast, no TF32, no CUDA/GPU, no batching/padding/vectorization across views.

The active-control ABI is frozen as the prebuilt native extension pams.temporac._fp_control. Its exact wheel/ELF bytes are environment-indexed and trace-loaded. It exposes only fe_tonearest, get_round/set_round, and get_mxcsr/set_mxcsr with the exact signatures in the JSON mirror. It is built before P06 and cannot be generated, compiled, JITed, or injected during replay. The CPU replay platform is exactly Linux x86_64 little-endian; any other platform fails closed.

The denormal probes use exact uint32 bit patterns, reinterpret them as CPU float32, perform one Torch scalar multiply, and compare exact output bits:

- 0x00800000 * 0x3f000000 must produce 0x00400000;
- 0x00400000 * 0x40000000 must produce 0x00800000.

Order is output-denormal then input-denormal. Run them before and after each numerical region. Pre-probe MXCSR is exact 0x00001f80; after pre-probes reset and reread exact 0x00001f80 before forward. Post-region control readback requires (MXCSR & 0xffffffc0)==0x00001f80; sticky status bits 0..5 may reflect arithmetic and are then cleared before any next region. FE mode and backend flags are reread around the same boundary. Unsupported, unreadable, warning-only, or mismatched state fails closed.

Each authority-bearing checkpoint tune evaluation uses one fresh subprocess for its 56 unbatched view calls, with no reuse across checkpoints. The consumed K7 evaluator is itself one fresh non-resumable process; it owns vault access and re-establishes/probes controls before each checkpoint replay region. These subprocess/process requirements add no training job or GPU hour.

## 6. A005-R4-B1/B2/B3 - executable native runtime and unique provenance bytes

### 6.1 Exact CPython 3.12.13 preinitialization and initialization

The authority supervisor embeds exactly CPython 3.12.13 on Linux x86_64 little-endian LP64. The profile freezes the exact six source files/hashes in Section 2 and the build predicates MS_WINDOWS=false, __CYGWIN__=false, Py_DEBUG=false, Py_STATS=false, WITH_PYMALLOC=true, _PyConfig_INIT_ISOLATED=3, PYMEM_ALLOCATOR_NOT_SET=0, PYMEM_ALLOCATOR_DEFAULT=1, and _PY_LONG_DEFAULT_MAX_STR_DIGITS=4300.

PyPreConfig_InitIsolatedConfig is the only constructor. Its ten Linux fields are listed in exact initconfig.h order and every constructor value is checked. The final preconfiguration changes only allocator from 0 to exact default token 1; parse_argv=0, isolated=1, use_environment=0, configure_locale/coercion/warn/UTF-8/dev all remain zero. Py_PreInitialize is called exactly once after the native audit hook and before any PyConfig setter or supervisor locale/decode/allocation path.

PyConfig_InitIsolatedConfig is the only configuration constructor. All 64 Linux fields are named in exact header order. The mirror gives all 64 constructor defaults and both complete 64-field final maps. Full and discovery profiles explicitly freeze use_frozen_modules=0, filesystem utf-8/surrogateescape, stdio utf-8/surrogateescape, exact orig_argv/argv, empty xoptions/warnoptions, int_max_str_digits=4300, every path input/output and prefix, importlib/main phase fields, and no command/module/filename.

The supervisor traverses the header exactly once: _config_init is checked and left constructor-owned; all 37 remaining scalar fields are directly assigned; every 21 owned string field is passed once through PyConfig_SetString, including exact NULL; every five owned list fields uses PyConfig_SetWideStringList, including length-zero/NULL-item lists. All ASCII strings widen one-to-one to wchar_t. PyConfig_SetBytesString, bytes argv, Py_DecodeLocale, public PyConfig_Read, implicit environment/path selection, and caller buffers are forbidden. Every PyStatus is checked immediately.

Py_InitializeFromConfig is called once. Its internal bound CPython configuration read may not change the selected final map. Before any bootstrap import, native access to the exact 3.12.13 runtime structs checks all ten preconfig and all 64 config fields, including dump_refs_file omitted by _PyConfig_AsDict; _Py_GetConfigsAsDict is fully canonicalized and hashed; sys.argv, sys.orig_argv, sys.path, sys.executable, filesystem encoding/errors, and all three stream encodings/errors are checked. stderr errors is exactly backslashreplace; stdin/stdout errors are surrogateescape.

PyConfig_Clear is called exactly once after successful readback and before bootstrap import. Any exceptional PyStatus fails at its exact stage, clears exactly once iff a PyConfig exists, closes authority FDs, and _exit(120); no Py_ExitStatusException, retry, fallback, or partial-state reuse is allowed. Joined normal exit invokes Py_FinalizeEx once and requires zero.

### 6.2 Exact Linux descriptor objects and three-stage readback

temporac.linux-fd-contract.v4 freezes numeric Linux x86_64 constants and a 20-key FD row:

access_mode, descriptor_flags_entry_u32, descriptor_flags_locked_u32, device_major_or_null, device_minor_or_null, eof_rule, fd, file_status_flags_u32, inheritable_after_lock, inheritable_at_exec_entry, initial_offset_or_null, object_type, payload_bytes_or_null, payload_sha256_or_null, pipe_capacity_bytes_or_null, queued_bytes_or_null, role, seals, seekable, size_bytes_or_null.

Full runtime has exactly fds 0..6: sealed read-only invocation, independent write-only /dev/null stdout/stderr, sealed read-only effective contract/source/environment, and write-only trace pipe. Discovery has exactly fds 0..3: sealed read-only capsule, independent /dev/null stdout/stderr, and write-only output pipe. Every listed fd enters exec inheritable with descriptor flags zero; native code locks exact FD_CLOEXEC and proves noninheritability.

A sealed memfd is populated, fully sealed, original writer closed, and reopened as a distinct O_RDONLY open-file description. It is regular, seekable, offset zero, exact size/hash, with exactly F_SEAL_GROW/F_SEAL_SEAL/F_SEAL_SHRINK/F_SEAL_WRITE and exact EOF by pread. Each /dev/null is a separately opened character device major 1/minor 3 with O_WRONLY|O_LARGEFILE, offset/st_size zero. Pipes are O_WRONLY, nonseekable ESPIPE, exact capacity, zero queued bytes at entry, one controller read peer, and terminal EOF only after the joined final writer closes.

temporac.linux-fd-readback.v4 has exact keys extra_fds, launch_instance_sha256, rows, schema, stage and exact 15-key typed rows. It records exec-entry, locked-pre-cpython, and post-pyinitialize. Full fds 3/4/5 are exact EBADF closed at both later stages; all retained descriptors remain byte/state equal. Discovery retains 0..3. extra_fds is exactly empty; the temporary /proc/self/fd enumeration descriptor is excluded only while enumerating itself and closed before serialization. Missing, reused, aliased, flag/offset/size/seal/EOF drift, or extra descriptor fails.

Each temporac.native-launch-instance.v4 binds exact argv, contract, cwd, launcher, mode, FD rows, and environment digest-or-null. Discovery requires null final-environment digest and the capsule layout; full runtime requires the exact final environment digest and seven-FD layout. The CPU environment commits the initialization/FD profile digests but never an invocation-specific launch digest, preventing self-hash.

### 6.3 Independent bootstrap-only generated-source discovery

temporac.generated-source-discovery-capsule.v4 is built before the final source/environment closure. It binds the accepted contract, exact supervisor path/bytes/hash, CPython interpreter/libpython and complete runtime-member index, an exact initial-native-image index, initialization/FD profiles, exact argv, cwd, envp, module paths, bootstrap_modules=[dataclasses], and the unique dataclasses runtime row:

- bytes 62,085;
- kind stdlib-source;
- normalized_path stdlib/dataclasses.py;
- provider_path lib/python3.12/dataclasses.py;
- SHA-256 d242aea5fcf6408b1c1f622442f88f68b9526ce1f8bd2890d74a144677c427d9.

The discovery argv is exactly the supervisor plus --discover-generated-v4. sys.path contains only stdlib and lib-dynload. envp is exactly LANG=C, LC_ALL=C, PYTHONHASHSEED=0, TZ=UTC. No project/site-packages path, source allowlist, installed project/wheel, trusted-generated index, final environment/code index, trace, G5a, receipt, data, or experiment artifact is present.

Before any discovery exec, P06 durably commits temporac.generated-source-discovery-precommit.v4 with exact capsule bytes/hash, contract, schema, and created_before_process_count=0. The same supervisor then runs twice in fresh discovery processes. Each imports exactly dataclasses as the sole requested module and allows only CPython-stdlib transitive imports. The hook captures the complete finite ordered sequence of every non-file-backed source compile/exec event; no caller dataclass application, pams/third-party import, dispatch, or final-index access is allowed.

temporac.generated-source-discovery-output.v4 has exactly capsule_sha256, rows, schema. Each row has the eleven exact generator/source fields in the mirror. Canonical output bytes and SHA-256 are recomputed; output_root=H(temporac.generated-source-discovery-root.v4,raw32(capsule_sha256),exact output bytes). The two outputs, hashes, roots, launch states, configuration readbacks, and FD readbacks must agree exactly. Equality is evidence, not authorization of the producer.

Only then is the trusted-generated index derived. Full runtime imports dataclasses first and consumes exactly that ordered stdlib-generated list, then imports pams and the ASCII runtime closure. Any project/third-party anonymous source, unseen generated bytes, or generated event after DISPATCH_BEGIN fails. After the trusted index and final environment exist, a distinct --embedded-cpython-v4 P06 invocation runs ordinary dispatch/receipt flow and binds the final trace. There is no discovery/final-environment cycle.

### 6.4 Installed wheel members and canonical association preimages

Every distribution wheel archive remains byte-bound. The installed-member index uses a sole normalize_rel_posix_ascii algorithm: strict printable ASCII, NFC unchanged, POSIX slash only, no NUL/backslash/colon/absolute/trailing/repeated slash or empty/dot/dotdot segment, and no case fold, percent decode, realpath, or separator rewrite. ZIP names with bit 0x800 decode strict UTF-8 and must then be ASCII; without it, every raw byte must already be printable ASCII and decodes one-to-one. Other encodings fail.

Every accepted installed row is byte-preserving member_tag=wheel-member. The installer-generated index exists with exactly zero rows: no console wrapper, RECORD rewrite, INSTALLER/direct_url, pyc, or other synthesis occurs. Prebuilt CPython/supervisor/loader/native runtime files are disjoint and belong to their own indexes.

repository_tag is an independent exact discriminator: repository-member requires nonnull project repository path/hash and exact installed/repository byte equality; no-repository-member requires both null. There is no inferred null state.

temporac.installed-member-association.v4 is an exact standalone 16-key object: bytes, distribution, generated_member_row_sha256_or_null, installed_path, kind, member_tag, normalized_member_path_or_null, repository_path_or_null, repository_sha256_or_null, repository_tag, schema, sha256, version, wheel_member_path_or_null, wheel_member_sha256_or_null, wheel_sha256. association_sha256 hashes its canonical compact sorted-key UTF-8 JSON plus exactly one LF. The installed index row repeats all 15 non-schema values by the same names and adds only association_sha256. Parse/rebuild/re-encode equality is mandatory; delimiter packing, tagged-H, renamed key, omitted null, pretty JSON, CRLF, normalized/raw mismatch, member swap, or repository relabel fails.

### 6.5 Typed executable origins and readback evidence

temporac.executable-origin-trace.v4 has exact event rows event, origin, origin_sha256, process_ordinal, sequence, sha256. The 16 event tokens include NATIVE_FD_READBACK and CPYTHON_CONFIG_READBACK as well as process/native/import/source/generated/dispatch events. Origins have no polymorphic string: each is one exact standalone canonical JSON-plus-LF object with schema temporac.executable-origin.<kind>.v4 and one of eleven closed kinds:

- cpython-core, cpython-extension, cpython-stdlib;
- installed-extension, installed-wheel, project-installed;
- native-image, process-image;
- trusted-generated;
- runtime-readback;
- dispatch-boundary.

Each kind's exact key list is in the mirror. Installed origins repeat distribution/version/member/wheel/installed/repository identity and resolve one association. Native/process origins repeat absolute path, tagged build-id null/present state, ELF hash, launcher/launch instance, provider class, and tagged installed-association null/present state. Generated origins repeat the complete discovery/trusted row identity. Readback origins repeat launch instance, profile, schema, stage, positive bytes, and complete hash.

origin_sha256 hashes exactly the embedded origin object's canonical bytes including LF. Event sha256 equals the exact digest projection fixed per event kind: executable source/provider/generated bytes, ELF image, readback bytes, or source allowlist at dispatch. Wrong kind/event/schema, missing/extra key, alternate encoding, tag/null swap, digest-only substitute, crosswalk mismatch, event projection mismatch, duplicate preimage, or parse/re-encode drift fails.

The native hook is installed before Py_PreInitialize. Sequence zero is PROCESS_IMAGE; exec-entry FD readback and initial images precede initialization; locked-preinit readback precedes Py_PreInitialize; post-init FD/config readbacks precede imports; the trusted sequence precedes DISPATCH_BEGIN; descendants use the same absolute supervisor with joined process ordinals. Ordinary artifact/data/receipt/index reads are excluded unless their bytes become executable. Trace remains post-run evidence and never feeds back into code, capsule, environment, G5a, capability, or its own authority.

### 6.6 Preserved runtime/science boundary

The new native/profile/capsule/association/origin schemas are provenance clarifications inside the existing P06/runtime boundary. They add no training or response job, GPU hour, algorithm, feature, teacher output, score term, identity, prediction, target member, threshold, metric, evaluator access, result, or claim. R3-B2's NATP exact14, 54-owner/106-edge DAG and all R2 scientific closures remain unchanged.

## 7. Preserved teacher/checkpoint/selection closure

The shared temporac.teacher-tune-input.v4 remains one deterministic NPZ for exactly 56 views: source 24..31 outer, block 0..6 inner, linear resampler, offset zero. It has N=25,680, E=25,624, seven exact members, 22,592,544 raw payload bytes, and a schema-specific 32 MiB cap:

| member | exact dtype/shape |
|---|---|
| analytic_chi | <f8[25624] |
| analytic_pulse | u1[25624] |
| sample_offsets | <i8[57] |
| sampled_source_clock | <i8[25680] |
| static_features | <f4[56,298] |
| teacher_input | <f4[25680,215] |
| traversal_bounds | <i4[56,10,2] |

Its exact nine-key receipt binds artifact/member bytes, code/config/contract/environment, and X0 manifest. It contains no natural or evaluator labels.

Every teacher checkpoint remains a deterministic uncompressed sorted NPZ with exactly the 16 current TempoRACTeacher state_dict members, each NPY v2.0 C-order <f4 with exact key/shape ledger in the JSON mirror. Its receipt remains exact 12 keys. The strict loader recomputes bytes/hashes, rejects extra/missing/wrong dtype/shape/order, instantiates a fresh CPU teacher, strict-loads, reserializes byte-identically, then sets eval/no-grad. teacher_sha256 is the exact artifact SHA only.

All 120 checkpoint artifacts/receipts and all 120 two-member tune outputs/13-key evaluation receipts remain available through K7. Each output has phase <f4[25680,2], reconstruction <f4[25680,149], raw payload 15,510,720, cap 16 MiB. The score index remains exact 120 rows and eight keys per row, with binary64 score bits independently recomputed and canonical +0.0. Selection precedence remains per seed (tune_objective,step) after the negative-edge filter, then global (tune_abstentions,mean reconstruction,tune objective,seed). Missing/tampered loser evidence fails.

K7 performs selection replay once globally after G5b PASS, not once per identity. Only that replay constructs the package-private typed SelectedTeacher.


## 8. A005-R3-B2 - satisfiable 54-owner inventory and uniquely typed DAG

temporac.pre-g5a-receipt-inventory.v4 still has exactly contract_sha256, rows, schema and exactly 54 rows in the immutable roster. Each row still has exactly node_class, owner, receipt_schema, receipt_sha256, upstream_owner_tokens, upstream_receipt_sha256, with equal-length ordered upstream arrays.

The class counts remain 17 pre-g5a-stage, 27 run, one G1, one K1, three X0I, one K3, one K4, and three NATP, total 54. The roster has exactly 106 direct upstream occurrences: 58 generic-stage, 30 run, three G1, two K1, three X0I, three K3, four K4, and three NATP. The canonical roster order and F23 precedence are unchanged.

The NATP schema is now satisfiable and explicit. temporac.natural-prediction-completion-receipt.v4 has exactly 14 keys:

- clean_seed_index_sha256;
- clean_seed_root_sha256;
- code_index_sha256;
- contract_sha256;
- drift_seed_index_sha256;
- drift_seed_root_sha256;
- environment_sha256;
- k6_receipt_sha256;
- owner;
- pilot_scope_manifest_sha256;
- schema;
- seed;
- status;
- upstream_receipt_sha256.

upstream_receipt_sha256 is an exact length-one lowercase digest array. Element zero must byte-equal k6_receipt_sha256, the NATP inventory row's sole upstream digest, and the unique K6-RESAMPLER receipt digest. owner/seed and both condition indexes/roots remain exact. A 13-key payload, redundant unequal scalar/array, wrong owner, or extra digest fails.

The DAG role set now includes k1.upstream. The K1 inventory row remains exact upstream owners [G0-ACQUIRE,G1-TEACHER-AGG]. It generates exactly:

- k1.upstream ordinal 0 from K1-COVERAGE to the unique G0-ACQUIRE receipt;
- k1.upstream ordinal 1 from K1-COVERAGE to the unique G1-TEACHER-AGG receipt.

K1's own exact nine-key receipt is unchanged; the receipt inventory is its canonical direct-dependency container. For ordinal zero, the exact G0 receipt's payload_index_sha256 must name a temporac.g0-acquisition-payload-index.v4 with exactly contract_sha256, population_manifest_bytes, population_manifest_sha256, schema. Those exact P402 manifest bytes/hash must equal the separately root-bound population manifest and k1-certificate-outcome-index.population_manifest_sha256, and every K1 population_row_sha256 must resolve to it.

For ordinal one, validate the exact G1 receipt and its selection, tune input, score index, checkpoint evidence, and exact three run receipts in seed order. Every nonnull K1 natural-input receipt must name that same G1 selection and selected checkpoint artifact/receipt; targets inherit the same association. ABSTAIN or early-no-output rows do not erase the direct G1 dependency.

Role generation is now exhaustive and has no catch-all:

- generic stages, X0I, K3, and K4 use stage.upstream;
- all 27 runs use run.upstream;
- G1 uses g1.teacher-run;
- K1 uses k1.upstream for exactly the two roster dependencies and keeps k1.feature/natural-input/target associations;
- NATP uses natp.k6 for its exact payload array and natp.prediction for 3,216 receipt associations per seed.

These rules generate all 106 roster-direct edges. Every edge points backward in the immutable roster, including K1 to G0/G1 and NATP to K6. A full topological sort must consume every typed node; missing/extra node, role, edge, ordinal, schema, payload equality, multiplicity, or cycle fails. G5a, its containing root, capability records, G5b, K7, metric/results, retries, future receipts, and acceptance reviews remain excluded, so no self-cycle is introduced.

## 9. A005-R2-B6 - scope/NATP completion and K6-to-prediction lineage

The canonical temporac.pilot-scope-manifest.v4 binds:

- protocol GT-bbox-assisted AlphaPose supplied-track pilot;
- labels partial-cache, GT-bbox-assisted, supplied-track;
- 110 train and 51 development videos;
- 268 train and 134 development identities;
- exact P402 population manifest;
- arms local/global/uniform/capacity-control;
- seeds 20260815/16/17;
- conditions natural-clean/natural-drift.

It contains no vault human count, period, density, metric, or main-paper eligibility.

Each NATP seed creates exact clean and drift temporac.prediction-seed-index.v4 artifacts. Each contains 402 identities x four arms = 1,608 rows in population then arm order for one fixed seed and condition. The seed root is H(temporac.prediction-seed-root.v4, condition ASCII, uint64_be(seed), exact index bytes).

temporac.natural-prediction-completion-index.v4 has exactly three rows in seed order. Each binds the one K6 receipt, pilot-scope digest, NATP receipt, both seed-index byte counts/hashes, and both seed roots. The top level binds exact scope bytes/hash, K6, both final condition indexes and roots.

For each condition, the existing 4,824-row prediction index is the exact merge of three seed indexes in population, arm, seed order. Every row is byte-equal to its source seed row. Thus six 1,608-row seed indexes yield exactly 9,648 cells, with no new prediction.

Each exact 14-key NATP receipt carries an ordered length-one upstream_receipt_sha256 whose element equals k6_receipt_sha256 and the unique K6 receipt; it creates natp.k6 ordinal zero. It also points to exactly 3,216 prediction receipts for its seed, clean rows first then drift rows, using natp.prediction ordinals 0..3215. G5a binds the completion index and all NATP nodes/edges. This proves the post-K6 freeze without adding a job, datum, metric, claim, or authority.

The G5a root-index class list is exact ASCII order with twelve classes: certified-development, checkpoint, executable-origin-trace, feature, k1-outcome, natural-prediction-completion, pre-g5a-receipt-inventory, prediction-clean, prediction-drift, score, target, teacher-checkpoint-evidence. Its receipt_root remains H(temporac.receipt-root.v4, exact root-index bytes). G5a's own exact 14-key schema is unchanged.

## 10. A005-R2-B7 - exact K1 row preimage

temporac.k1-certificate-outcome-index.v4 still has exactly 402 rows in population order and 13 exact row keys. It freezes eligibility, status/reasons, same identity/component/feature provenance, natural-input-or-null, and target pair-or-null before G5a. Cdev remains exactly the ordered val/CERTIFIED projection, subject to canonical K1 floors and retained minimum 108 identities/eight components. It is not all 402 and cannot be a later subset.

For a committed K1 row, k1_outcome_row_sha256 is now defined uniquely:

1. construct a standalone object containing exactly all 13 K1 row keys and the exact embedded values, including explicit nulls;
2. canonicalize as compact sorted-key UTF-8 JSON with no BOM, duplicate, or nonfinite value;
3. append exactly one terminal LF byte;
4. SHA-256 every byte including LF.

The standalone object must parse and recanonicalize byte-identically to the embedded row. Tagged H, parent/index bytes, ordinal, pretty JSON, omitted null, CRLF, or LF-excluded digest is forbidden. Every certified-development row carries this exact digest, closing Cdev-to-K1 provenance.

K2 continues to publish only the exact target bytes already hashed at K1. The seven-member target NPZ and ten-key target receipt remain unchanged.

## 11. Preserved natural inference and K7 target replay

The natural builder accepts a validated typed feature, exact feature artifact/receipt, and typed SelectedTeacher. It has no caller selected_teacher_sha256, phase, reconstruction, static features, traversal bounds, module, state_dict, or digest-only path.

The unique order remains validation, canonical preprocess, per-run geometry-only integer landmarks, early LANDMARK, consecutive-landmark bounds, normative F5, one CPU teacher forward per run, output cast/hash, complete certificate. A run with fewer than three landmarks abstains before any teacher output or target artifact.

F5 uses continuous[149] and edge lengths in exact binary64 traversal/subedge Neumaier order, then performs one C-order <f4[298] cast. teacher_input is exact <f4[T,215]. Phase and reconstruction are computed in CPU float32 and then each whole C-order output is cast once to <f8[T,2] and <f8[T,149] and hashed. Caller injection is impossible.

For each Cdev row, after one global selection replay K7 reloads the exact feature, reruns natural inference/certification, rebuilds the unchanged seven-member target and exact ten-key receipt, and requires bytewise equality of both. Digest-only acceptance fails. No target receipt schema amendment is required.

## 12. Capability order and evaluator population

Pre-consume verifies only immutable non-vault commitments: effective contract, source/environment locks, P2-METRIC/S0 binding, scope manifest, P402, all indexes/roots/DAG, 54-owner inventory, K6/NATP completion, exact 9,648 predictions, K1/Cdev evidence, target counts, G5a/grant/ledger, and invocation_count=1.

Before consumption it may not open, stat, hash, enumerate, deserialize, map, or acquire a vault handle. Supplied vault rows or caller cardinality are not inputs. A non-vault failure does not consume and does not touch the vault.

At fresh non-resumable evaluator-process start, consume the one-use grant atomically and durably. Only that same process opens the vault, validates exact root/join/P402 bijection/manifest component, positive integer count and period units, and derives cardinality exactly 402. Failure burns the grant and forbids retry, resume, restart, or replacement. K7 starts only after G5b PASS in that process.

Metrics still cover all 134 development identities including zero-estimate stubs. Cdev controls certificate/target evidence only and never shrinks evaluator population.

## 13. F4 direct Mapping composition remains closed

The builder may return MappingProxyType. The consumer first exhausts exactly one items pass into a plain six-key dict snapshot, validates and canonicalizes that same snapshot, and never rereads caller values. A guarded second items pass only detects stateful changes and must match exact string pairs. Direct MappingProxy composition works; malformed or state-changing Mapping fails TOCTOU. No dict() requirement is imposed on the caller.

## 14. Required future tests and preserved closed semantics

Round 5 binds all 57 current non-self paths described in Section 2. No test, review, implementation, launch, or experiment was executed during this authoring revision.

All earlier positive/negative contracts remain mandatory: normalized SmoothL1, one-run certificate bounds, three-run arrays, FE/MXCSR/denormal/backend locks, all 120 checkpoints/outputs/evaluations, exact 54 owners and 106 direct edges, NATP/K6 prediction lineage, K1 row preimages, K7 replay, vault order, and F4 Mapping composition.

A005-R4-B1 tests must independently enumerate all ten PyPreConfig and 64 PyConfig fields against the exact header/default/final maps; attack either constructor, build predicate, field, setter count/order/value/null/list ownership, public/internal read behavior, orig_argv/path/encoding/use_frozen_modules/main/importlib phase, PyStatus branch, clear/finalize, or readback. FD attacks cover class, access/status/descriptor flags, inheritability, offset, seekability, size, seals, EOF, /dev/null device, pipe capacity/queue/peer, shared description, fd3/4/5 lifetime, reuse, and extras at each of three stages. Exact full and discovery positive launches must pass all native/config/stream/FD readbacks.

A005-R4-B2 attacks change capsule CPython/dataclasses/native/init/FD bytes, argv, envp, cwd, path, bootstrap action, precommit ordering/count, either discovery output, output root, or final-run separation; import pams/site-packages, apply a caller dataclass, read final code/environment, dispatch, truncate/filter/reorder source, learn a row from final execution, or let output authorize its producer. Positives create the zero-process precommit, run two fresh capsule processes with byte-identical complete outputs/roots, derive the trusted index, then run one distinct final P06 supervisor that consumes it under the final environment.

A005-R4-B3 attacks the UTF-8-flag/ASCII branch, normalized/raw relation, wheel member/hash, member/repository discriminator, project source equality, forbidden installer-generated row, association key/type/null/terminal-LF encoding, origin event-kind/schema/key/path/build-id/native/generated/readback identity, launch instance, event digest projection, parse/re-encode equality, uniqueness, or trace membership. Positives cover a valid ASCII ZIP member without bit 0x800, exact project installed/repository equality, all eleven origin kinds at schema level, all three FD readback stages, config readback, and direct trace/association reconstruction.

All previously closed A005-R2-B1/B2/B5/B7/B8, A005-R3-B2, F4, natural inference, teacher selection, target schema, K1/Cdev science, prediction cardinality, capability order, data firewall, thresholds, and evaluator estimands remain unchanged. The complete machine-readable test contract is mirrored in Appendix A.

## 15. Closure matrix

| finding | Round-5 closure |
|---|---|
| A005-R4-B1 | exact CPython 3.12.13 isolated constructor/default/final/setter/internal-read contract for all 10+64 Linux fields; exact FD objects and three typed readback stages |
| A005-R4-B2 | independent no-project dataclasses capsule; durable zero-process precommit; two byte-identical complete output/root discoveries; then trusted/final environment and a distinct full P06 invocation |
| A005-R4-B3 | sole ZIP/path normalization; member and repository discriminants; exact 16-key association canonical JSON+LF; eleven closed origin schemas, typed readbacks and exact event digest projections |
| A005-R3-B2 | unchanged NATP exact14, K1 G0/G1 roles, 54 owners, 106 direct backward edges, K6 lineage and G5a typed root |
| Previously closed science/interfaces | unchanged: teacher/tune/score/natural/K1/Cdev/target/prediction/capability/F4 semantics, 27 jobs, 93 hours, data, thresholds and claims |

Every row is PROPOSED_PENDING_ROUND5_REVIEW, not accepted authority.

## 16. Final disposition

This amendment remains same-family provisional with every authority value zero. It preserves the rejected Round-4 fixed pair byte-identically at timestamp 20260816_173526, creates an exact Round-5 timestamped pair at 20260816_173527, and uses fixed names only as byte-identical aliases. A fresh independent Round-5 review is required before effective-contract construction or implementation.

This revision changes no source, test, proposal, plan, tracker, review, MANIFEST, trace, server, data, sealed, heldout, vault, Git, checkpoint, prediction, target, evaluator, training, result, threshold, job, or claim. It starts no review, test, experiment, server task, or capability action.

## Appendix A - complete machine-readable mirror

The following fenced payload is byte-for-byte the standalone Round-5 JSON file, including its single terminal LF before the closing fence.

```json
{
  "amendment_id": "TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE",
  "amendment_round": 5,
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
    "current_runtime_bootstrap_snapshot_non_authorizing": {
      "pyproject.toml": "14b3b35bbfa57fc982e24ef2f999df288893c0f0d12a53a9f82120babefead38",
      "src/pams/__init__.py": "8b6ffbf3ac8ed7e43f1f40ef5793baa087f73b303eff342d891380bc9a5d67ce",
      "src/pams/types.py": "78c2ad53e98856425492a1711faa62fc2bb7764767dc48a22f7c4d5e043a3e31"
    },
    "current_source_snapshot_non_authorizing": {
      "src/pams/temporac/__init__.py": "e3f71c2d85974ac3f635f76ee61f8e1f793b9bc251b25ff4336563d4da52c2a3",
      "src/pams/temporac/certify.py": "468ce8418bff0f43582d341f3e1b8064cf1c429301d1f56be84b8e13ad59a3f1",
      "src/pams/temporac/contract.py": "5ab8fb62dc558ba78e95fc07e52c1f1a1afbff5b1c5003834079eea6143d745f",
      "src/pams/temporac/cue.py": "702784bef27db94c1dc027d036a64d7461ae618613baec5e68061254f98569c7",
      "src/pams/temporac/decode.py": "766cd6ca4b9e133041bd602f50fde840ec7caadd376bd6851799583d44f31f5e",
      "src/pams/temporac/evaluator.py": "f981c6c7061c3388f06ef39466ef98192e65c62c501821be2b211f903cfaf9de",
      "src/pams/temporac/feature_io.py": "f54f17ca204c242d304314f4280e7540c28588e86ea588a405fff02176b95428",
      "src/pams/temporac/fixtures.py": "979b8cee4fed117f5471cf399794f615c60eae52dbba2c946109ff836206b464",
      "src/pams/temporac/gates.py": "b6c9d7432443d7ad36738a81700746709fd571d475514aa6f1bbcd39be744b4a",
      "src/pams/temporac/hashio.py": "aa460d5de51f107b2fc6b2ad171747e0da38a87cf30f82a869a335c268f1d613",
      "src/pams/temporac/metrics.py": "1dadf08e89330bed2ab773afda05bffcb3fe2252b0d3e23c0bcc18c2c3b049dd",
      "src/pams/temporac/nola.py": "ad6a13203f09c13a37a854333d4458ca8ce5a03bb186c0946f014a4e8cb9a637",
      "src/pams/temporac/objective.py": "09a77231a48a869cbc83e0aa1aa2502f86f4abbcbab8c84f2b2c5abbd729ac87",
      "src/pams/temporac/prediction.py": "7c57725240e83b7c06227a3b27e5f3cadb4590170488e5e126180caba8c13808",
      "src/pams/temporac/preprocess.py": "042b5cae1917359cb5596837c81c0e36658ff99a6a8fd6f2602b478ca6f98bc3",
      "src/pams/temporac/quadrature.py": "44f4b91dc4d5631442982732177a6f57726c0b8f42594380b5a75f2209efc2f3",
      "src/pams/temporac/receipts.py": "564039b5d37a4a2759544a9ae65f267b57d7f6c1e46bca812ae5ed4f4a2c17e4",
      "src/pams/temporac/response.py": "04df32d6e2e4061d86d813f611fb9ff277f2be8312af4efbc7219785af02f9bd",
      "src/pams/temporac/runtime.py": "9421661046ee686508607529ed972b087e724c2c3a5c487e220c2f8e4038a0c8",
      "src/pams/temporac/teacher.py": "30575fa64875aa6b77de157cf6af348b3b7900851a1c1742993842eea7928b30",
      "src/pams/temporac/training.py": "0af8ecbadfcda43d2b9f60853c99210b316786ff4310ed5d0a32d9e366f627a4",
      "src/pams/temporac/trusted_packer.py": "40d6e3d5be1b26b7ba0f0f7aadf8da7496c95c4e829face5c7e9faaee2cdf94e",
      "src/pams/temporac/types.py": "4249eac2738f5b2edf53f38cbb681dffae609c151d493dd03b625fe10ad4c3ca",
      "src/pams/temporac/x0.py": "5f4585425a83bf63779265db9004d210c5c2875a579ebc9a6c51297eb647c7a2"
    },
    "current_test_snapshot_non_authorizing": {
      "tests/temporac/__init__.py": "dd07192e3b73d582dfb5f0325cfafd7462040c76574ad8241a661fdc4713a24e",
      "tests/temporac/test_certificate_artifact.py": "f2b0ef6e6d977bf3532595147fba89b611258063ee5e36e30657c31c62e56c20",
      "tests/temporac/test_certificate_tau.py": "f7e2f25ab93226568910b856167e441037213ba3dd937d6915587037b7c6a94b",
      "tests/temporac/test_contract_hashio.py": "524c9ddf04e8cd29735c2c7d958f5f9e31afd8fea0bb3b6068a09b2de75b7104",
      "tests/temporac/test_feature_io_firewall.py": "f049e54c7254a848dc6cdd10afb853c5538fd8ecabd02b86950eb77d8a74c12b",
      "tests/temporac/test_gates_fixtures.py": "d6d91a7dd606c42f672bb015d77c5c10406c2c6de69ed3a8bff76a0ce4f713d3",
      "tests/temporac/test_metrics_evaluator.py": "a6f4aa275796b06162cac863674205d85b07adc46f89bc8d9821c48d17440fc6",
      "tests/temporac/test_nola_decode.py": "80586866a9293e892c2cab1ca7318b97d341c2aa5990f7e1a31695f18376ff0b",
      "tests/temporac/test_objective_runtime.py": "cc3685e2bf53ed7535addb1c09f08e0520dbd9143ba5d44da1cca28761107fa5",
      "tests/temporac/test_operator_manifest.py": "7f2c90c320d7378a74fcc64f6c808eb6e426cdcfbd73cab0fb1242f690db9db0",
      "tests/temporac/test_prediction_identity.py": "7967b11ee6585c2664fa4f39f5a9d9798ab407317f0d4efcf790f8d7a7bd5148",
      "tests/temporac/test_prediction_receipts.py": "25fe8cc8696ab6c4108ba909b2f315c83b674fbe6ec9a770e6b7f6e449b410d4",
      "tests/temporac/test_preprocess_geometry.py": "2edda5c593992c32aca30452340fce357930262978e331d37b6e5d55d3eb3b3f",
      "tests/temporac/test_quadrature_cue.py": "4b7f977cbd31dc441ac021666bbc036b1d7de505ee884445df356b57b88374b6",
      "tests/temporac/test_response_invariants.py": "b2e88e011bb06b9e21011ff252b630ffcbfcbe53a29638f8983b563f00ffae63",
      "tests/temporac/test_teacher_certificate.py": "a5dc0bff236f417ab570193e46542b693d0f201fedd91693e6c572a70d20e012",
      "tests/temporac/test_training_fresh_graph.py": "52b9f266173fec04152240d629dbf651aadd5b85d5839abcfe9f16fedc608845",
      "tests/temporac/test_x0.py": "90206f53c7fffeabe9bcb3d6374fd60d3c51c69a7d1137df0cba797b583d1ef8",
      "tests/temporac/test_x0_manifest.py": "074bb8df9dce5d1ea9e987499dac0d8824e1803a5f4bdc32495bd957e8c4c6a3"
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
      "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND4.md": "988eb7c2b61eb445a07e1c17b7bcf44be256f9948d6600a6ed6edc62ef71cfc1"
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
    "digest_binding": "environment_sha256 hashes complete exact environment bytes, including floating-point control, CPython initialization profile, FD contract (which contains the readback schema), native launcher, discovery capsule/precommit/output/trusted-source digests, and installed-distribution/CPython/wheel/ELF indexes; it agrees in every checkpoint, tune, selection, G1, G5a and K7 object.",
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
          "F_GETFD",
          "F_GETFL",
          "F_GETPIPE_SZ",
          "F_GET_SEALS",
          "F_SEAL_GROW",
          "F_SEAL_SEAL",
          "F_SEAL_SHRINK",
          "F_SEAL_WRITE",
          "FD_CLOEXEC",
          "O_ACCMODE",
          "O_APPEND",
          "O_ASYNC",
          "O_DIRECT",
          "O_DSYNC",
          "O_LARGEFILE",
          "O_NONBLOCK",
          "O_PATH",
          "O_RDONLY",
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
          "schema"
        ],
        "digest_rule": "launch_instance_sha256 is SHA-256 of the complete canonical launch-instance bytes including LF. The CPU environment contains only this closed schema/profile, never an invocation-specific payload digest; each post-environment launch instance may bind environment_sha256 without a self-cycle.",
        "mode_tokens": [
          "bootstrap-generated-discovery",
          "full-runtime"
        ],
        "mode_values": {
          "bootstrap-generated-discovery": "argv equals the exact discovery capsule argv, cwd '/', environment_sha256_or_null is null, fd_rows are exactly discovery_layout 0..3, and launcher/contract/capsule payloads equal the capsule/precommit",
          "full-runtime": "argv equals runtime_launcher.argv, cwd '/', environment_sha256_or_null is the exact final environment digest, fd_rows are exactly full_runtime_layout 0..6, and launcher/contract/source/environment payloads match"
        },
        "schema": "temporac.native-launch-instance.v4"
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
        "A sealed-memfd is a regular anonymous memfd created writable, populated, fully sealed, closed, and reopened through its descriptor as a distinct O_RDONLY open-file description before exec. file_status_flags_u32 equals O_RDONLY|O_LARGEFILE; seekable=true; initial offset=0; size=payload_bytes; payload hash covers exactly size bytes; seals are exactly the four value-domain tokens; pread(size) then pread(one byte at size) proves exact EOF while offset remains zero.",
        "FD1 and FD2 are independently opened /dev/null character devices with major 1 minor 3, O_WRONLY|O_LARGEFILE, seekable=true, offset zero, st_size zero, no seals/payload/pipe fields, and EOF token not-applicable-character-sink. Sharing or duplicating one open-file description is forbidden.",
        "A pipe-write-end has O_WRONLY and no append/nonblock/async/direct/sync/path bit; seekable=false with ESPIPE and null offset/size/seals/payload/device fields; exact positive F_GETPIPE_SZ, FIONREAD queued bytes zero at entry, exactly one controller-held read peer, and EOF token joined-reader-eof-after-final-writer-close.",
        "The controller closes every fd outside the exact mode layout before absolute-path exec. The supervisor records exec-entry readback, sets FD_CLOEXEC, preads and validates sealed inputs, closes full-runtime fds3/4/5, and records locked-pre-cpython readback. It retains full-runtime 0/1/2/6 and discovery 0/1/2/3. Immediately after Py_InitializeFromConfig it records post-pyinitialize readback; retained state must be unchanged and every required closed fd still EBADF.",
        "Each child exec constructs a fresh launch instance and descriptor table. No inherited duplicate, fork-without-exec, caller fd, socket, tty, disk regular file, writable input memfd, shared open-file description, changed offset/size/seal/flag, early EOF, unread output peer, extra fd, or missing readback is permitted."
      ],
      "schema": "temporac.linux-fd-contract.v4",
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
      "acyclic bootstrap-only dataclasses discovery capsule/precommit/two equal outputs/root and exact trusted generated rows",
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
        "Post-initialize complete struct/_Py_GetConfigsAsDict/sys-stream and FD readbacks emit CPYTHON_CONFIG_READBACK and NATIVE_FD_READBACK and must match before PyConfig_Clear and authority imports. Full runtime then imports exact closure, proves no generated event, emits DISPATCH_BEGIN and dispatches.",
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
      "completeness": "Rows are exactly the trusted projection of the two byte-identical bootstrap-only dataclasses discovery outputs. Every full-runtime process must reproduce/consume this finite ordered stdlib-generated list before DISPATCH_BEGIN and observe no other anonymous generated event. Project/site-packages import in discovery, caller dataclass generation, missing/extra/reordered/different output, row learned from final runtime, or any generated event after DISPATCH_BEGIN fails.",
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
        "output": {
          "exact_bytes_rule": "Each discovery serializes exactly capsule_sha256, the complete rows sequence, and schema as canonical compact sorted-key UTF-8 JSON plus one LF. The controller recomputes output_sha256 and the tagged output root; the two fresh outputs must be byte-identical. No placeholder/guessed output digest is precommitted.",
          "expected_rows": "rows is the complete finite ordered sequence of every non-file-backed source compile/exec event caused solely by importing exact dataclasses and its CPython-stdlib transitive imports under the capsule. The sequence may be empty or nonempty only as freshly observed; it cannot be predicted, caller-supplied, truncated, filtered by qualname, or expanded by any other action.",
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
          "Launch the same supervisor twice in fresh processes with exact --discover-generated-v4 argv, four-FD discovery layout, literal envp/cwd and discovery PyConfig. The native hook precedes Py_PreInitialize. Import exactly dataclasses as the sole requested module and allow only its CPython-stdlib transitive imports; project/site-packages import, dataclass application to a caller class, final environment/code/trusted-index access, dispatch, data, receipt or artifact access fails.",
          "Capture the complete finite non-file-backed source compile/exec sequence into the closed output. Require both canonical output bytes, hashes and tagged roots byte-identical and bind both launch/config/FD readbacks before deriving trusted rows.",
          "The future full-runtime source closure may reproduce only these exact stdlib-generated rows before DISPATCH_BEGIN and no anonymous generated row afterwards. Current source is non-authorizing and may require separately reviewed refactoring that predefines project dataclass-generated methods; silently learning rows from project/final imports is forbidden.",
          "Only after capsule/precommit/two outputs/trusted index are fixed may the final CPU environment be serialized. Execute a distinct full P06 invocation under --embedded-cpython-v4; validate final source/environment, consume every trusted row exactly once in order, follow ordinary dispatch/receipt, and bind launch/readback/trace evidence. Discovery output never authorizes a process that produced it."
        ],
        "schema": "temporac.generated-source-discovery-contract.v4"
      },
      "full_runtime_order": [
        "Immediately after initialization/readbacks and before importing pams, import exact dataclasses once under profile bootstrap-only-dataclasses and consume the complete trusted generated row sequence.",
        "Then import pams, pams.types, pams.temporac, protocol_entry, and the remaining ASCII source-allowlist runtime module order. dataclasses is already cached; any project/third-party anonymous generation during these imports fails.",
        "After exact closure validation emit DISPATCH_BEGIN and dispatch. No generated event is accepted after the trusted bootstrap sequence."
      ],
      "generated_source_encoding": "The audit hook receives str or bytes. str is encoded once as strict UTF-8 with no BOM; bytes are used verbatim. generated_source_hex is lowercase hex of those exact bytes, source_bytes is its positive length, and generated_source_sha256 hashes them. NUL, undecodable filename, code-object-only execution, marshal/pickle code, or source over 1 MiB fails.",
      "generator_origin_class_tokens": [
        "cpython-stdlib-source"
      ],
      "kind_token": "trusted-bootstrap-generated-source",
      "mandatory_runtime_classification": "Only exact CPython-3.12.13 stdlib generation caused by the capsule's dataclasses import is admitted. Generator source/hash/qualname, compile filename/mode, generated bytes, profile and ordinal all match one output row. Project/third-party generator identity and unseen output are never authorized.",
      "noncircularity": "Initialization/FD/CPython/native inputs precede the capsule and its zero-process precommit; two discoveries produce output; output produces this trusted index; only then may the final code/environment and full supervisor run. The capsule has no final environment/code/trusted-index/trace digest, and the final environment refers forward to capsule/precommit/output/trusted digests. No discovery result authorizes the process that produced it.",
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
      "schema": "temporac.trusted-generated-source-index.v4",
      "top_level_closed_keys": [
        "contract_sha256",
        "cpython_runtime_member_index_sha256",
        "discovery_capsule_bytes",
        "discovery_capsule_sha256",
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
    "final_status": "PROPOSED_PENDING_ROUND5_REVIEW",
    "implementation_required_after_acceptance": "Only after an independent Round5 ACCEPT may a separately authorized implementation/review create the effective-contract index, complete CPython initialization profile, discovery precommit/capsule/output, typed installed-member associations, populated CPU environment, native supervisor and launch instances, executable-origin traces, artifacts, receipts, indexes, tests, and replay evidence. Current source/tests remain a bound non-authorizing snapshot.",
    "no_actions_taken": [
      "no source-code or test-file change and no test execution",
      "no plan, tracker, proposal, review, MANIFEST, trace, or Git change",
      "no server, data, sealed, heldout, vault, result, checkpoint, prediction, target, evaluator, training, or experiment access",
      "no authority, launch, gate, claim, result, job, threshold, data, or method expansion"
    ],
    "review_required": "A fresh independent Round5 specification review must inspect the exact timestamped MD/JSON pair and verify A005-R4-B1, A005-R4-B2, and A005-R4-B3 while confirming A005-R3-B2 and every already closed scientific/interface semantic remain unchanged before any implementation, acceptance-derived authority, or execution."
  },
  "effective_contract": {
    "circularity_policy": "No placeholder, null, guessed value, amendment self-hash, accepted-review self-reference, or fixpoint is permitted.",
    "closed_keys": [
      "rows",
      "schema"
    ],
    "construction": [
      "The index does not exist until an independent fresh Round5 review accepts this exact timestamped amendment pair.",
      "Each row binds the positive exact byte count and SHA-256 of the complete named bytes; rows occur only in the listed role order.",
      "contract_sha256 is SHA-256 of the complete canonical effective-contract index bytes, including LF.",
      "The index itself has no contract_sha256 field. None of the five constituent documents may contain the resulting digest.",
      "Every later artifact, receipt, index, source allowlist, environment lock, target contract member, G5a field, and K7 replay uses this one digest; the old proposal digest alone is invalid after acceptance.",
      "The mutable fixed-name amendment aliases, rejected archives, prior reviews, and reviewer traces are never effective-contract constituents."
    ],
    "current_value": "UNASSIGNED_UNTIL_ACCEPTED_REVIEW",
    "role_order": [
      {
        "path": "refine-logs/temporac/FINAL_PROPOSAL.md",
        "role": "canonical_proposal"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_173527.md",
        "role": "amendment_markdown"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260816_173527.json",
        "role": "amendment_json"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND5.md",
        "role": "accepted_review_markdown"
      },
      {
        "path": "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND5.json",
        "role": "accepted_review_json"
      }
    ],
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
        "PY_CODE_EVAL": "origin kind matches the already emitted source/generated provenance; raw code-object-only eval fails",
        "PY_CODE_EXEC": "origin kind matches exact cpython/project/wheel/generated source; raw or unmatched code object fails",
        "PY_EXTENSION_LOAD": "origin kind cpython-extension or installed-extension and exact ELF/crosswalk membership",
        "PY_FROZEN_IMPORT": "origin kind cpython-core with provider_kind frozen-provider",
        "PY_IMPORT_SOURCE": "origin kind cpython-stdlib, installed-wheel, or project-installed; complete source bytes/hash and crosswalk exact",
        "PY_SOURCE_COMPILE": "same typed origin and digest as the unique preceding source event",
        "PY_TRUSTED_GENERATED_COMPILE": "origin kind trusted-generated; complete discovery/root/generator/source identity equals one trusted row",
        "PY_TRUSTED_GENERATED_EXEC": "same trusted-generated origin as the immediately preceding compile; no generated event after DISPATCH_BEGIN"
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
          "project-installed": "installed_member_sha256",
          "trusted-generated": "generated_source_sha256"
        },
        "PY_CODE_EXEC": {
          "cpython-core": "provider_sha256",
          "cpython-stdlib": "runtime_member_sha256",
          "installed-wheel": "installed_member_sha256",
          "project-installed": "installed_member_sha256",
          "trusted-generated": "generated_source_sha256"
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
      "membership": "Every event resolves to exactly one precommitted authority row and one event-specific origin preimage. Installed origins repeat and equal the exact association preimage; CPython origins equal runtime rows; generated origins equal discovery-output/trusted rows; native/process origins equal launcher/ELF. Trace and launch instance are evidence only and cannot add or mutate authority.",
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
            "project-installed",
            "trusted-generated"
          ],
          "PY_CODE_EXEC": [
            "cpython-core",
            "cpython-stdlib",
            "installed-wheel",
            "project-installed",
            "trusted-generated"
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
          "trusted generated": "trusted set is empty in this profile, so this kind is structurally defined but any instance is rejected"
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
    "noncircularity": "Effective contract precedes CPython/native/init/FD/source/wheel inputs; the bootstrap capsule and zero-process precommit precede both discoveries; equal output precedes trusted-generated rows; those plus code/install indexes precede final environment; launch instances and typed traces follow environment. No capsule depends on final environment/code/trusted/trace, and no trace or association feeds back into precommit authority.",
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
  "generated_at": "2026-08-16T17:35:27+08:00",
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
      },
      {
        "path": ".aris/traces/research-refine/20260816_temporac_certificate_amendment005_review_sol/run.meta.json",
        "sha256": "2b086b41942ef8badc2dfbd31a993630be2620b4bb15033a4bfd52213533a013"
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
      "A005-R4-B3": "sole canonical association and event-specific origin preimages with exact normalization, discriminants, all wheel/repository/native/generated identities, equality and uniqueness"
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
      "free-form/digest-only trace origin, wrong event-kind schema, missing/extra identity key, alternate origin encoding, association/origin digest swap, path normalization alias, native build-id mismatch, generated row mismatch, or non-gap-free duplicate event order"
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
      "all 57 bound current paths rehash exactly: 24 TempoRAC sources, 19 tests, 11 planning/review files, and three runtime-bootstrap files, with fixed/timestamped amendment outputs excluded from self-binding",
      "exact CPython 3.12.13 release constructor maps contain all 10 Linux PyPreConfig and 64 Linux PyConfig fields; both mode profiles survive native/runtime/sys-stream readback, every setter/status/clear edge is observed, and FD mock entry/lock/post-init states match",
      "a precommitted no-project dataclasses capsule launched twice produces identical canonical output bytes/root; only afterward the trusted index/environment are built and a distinct final P06 dispatch consumes the exact rows",
      "present and generated installed-member fixtures reconstruct byte-identical association preimages, and every trace token reconstructs one event-specific origin object with exact wheel/repository/native/generated equality"
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
    "status": "SPECIFIED_NOT_RUN; no test execution is authorized by this amendment"
  },
  "round5_closure_matrix": {
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
    "preserved_closed_semantics": {
      "closure": "UNCHANGED",
      "rule": "all teacher/tune/score/natural/K1/Cdev/target/prediction/capability/F4 science, 27 jobs, 93 hours, data, thresholds, G5a/target schemas, and claims remain unchanged"
    }
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
  "status": "PROPOSED_PENDING_ROUND5_REVIEW",
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
