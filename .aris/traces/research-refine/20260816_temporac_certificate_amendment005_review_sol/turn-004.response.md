# TempoRAC Contract Amendment 005 Certificate Provenance - Round 4 Normative Review

Date: 2026-08-16  
Reviewer: `/root/temporac_certificate_amendment005_review`  
Route: Codex `gpt-5.6-sol`, OpenAI same-family, ultra review tier  
Assurance: provisional; not cross-family independent  
Verdict: **REVISE**  
Blocking groups: **3**  
Authority: **none**

## Outcome

The Round-4 candidate materially repairs both findings from the Round-3
review. In particular, it adds an installed-distribution member ledger, a
CPython runtime-member ledger, an absolute embedded-CPython supervisor whose
native audit hook precedes package import, a trusted-generated-source ledger,
an exact fourteen-key NATP receipt, and the two ordered `k1.upstream` edges.
The 54-owner receipt roster and its 106 direct dependency occurrences are now
internally satisfiable.

It is nevertheless not yet an executable byte-unique provenance contract.
Three authority-bearing defects remain in the new R3-B1 repair:

1. the purported exact embedded-CPython launch omits the preconfiguration and
   most real `PyConfig` state, and its file-descriptor contract is not an exact
   OS descriptor state;
2. the generated-source index is defined as the result of P06 executions that
   already require that same index and the containing final environment; and
3. two new association encodings - installed wheel-member crosswalk rows and
   post-run trace origins - do not have unique typed byte preimages.

Each defect permits two inequivalent implementations or evidence objects to
satisfy the written fields, or makes the stipulated construction temporally
impossible. Exact supervisor, environment, and trace digests can record which
choice an implementation happened to make, but the amendment supplies no
normative rule by which a verifier can accept one choice and reject the other.
The verdict is therefore `REVISE`, not conditional acceptance.

## Reviewed bytes and terminal integrity

The exact inputs matched at the start of review and at terminal rehash:

| Input | Bytes | SHA-256 |
|---|---:|---|
| timestamped Round-4 Markdown `_20260816_154812.md` | 203,317 | `0552aaca74b34929ead1124cce846d2a583931aa89ad29fcf0f97486b7be0198` |
| timestamped Round-4 JSON `_20260816_154812.json` | 165,291 | `04d24ad19b19470bcd675974766a279dc7691d3fdaecf665e3fd24e4a4aaeb82` |
| fixed Markdown alias | 203,317 | `0552aaca74b34929ead1124cce846d2a583931aa89ad29fcf0f97486b7be0198` |
| fixed JSON alias | 165,291 | `04d24ad19b19470bcd675974766a279dc7691d3fdaecf665e3fd24e4a4aaeb82` |
| archived rejected Round-3 Markdown `_20260816_143304.md` | 166,625 | `8d02c70186c38959c74b774e211427f86435664d00106159c0da11ce5843dae9` |
| archived rejected Round-3 JSON `_20260816_143304.json` | 138,394 | `187aa3450a9e40f1c75a04b135c5e97e856e18f0165e677a062317e3587be7f2` |
| Round-3 review Markdown | 16,188 | `e29f1cab1e92feba575cf7d5f446de4f8388b42b7c2d1c65e21d8187f856a54e` |
| Round-3 review JSON | 14,826 | `df7917f5eb215779afca36a4db1b978f0a2f6b64b720267b2b14cb20e703339a` |
| canonical proposal | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| experiment plan | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| experiment tracker | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |
| certificate postfix Round-2 Markdown | 12,283 | `93a9cae771b34c185b9057286794b894d615aec5424011df145aaf8900f008d8` |
| certificate postfix Round-2 JSON | 14,527 | `144d2f22f4f75166b8cbefe54157d08690408979dd7cb34ef8c97e1738432872` |

The fixed aliases are byte-identical to the `_20260816_154812` pair. The old
`_20260816_143304` archive remains byte-identical to the rejected Round-3
candidate and was not confused with the current pair. The current Markdown
Appendix A is byte-identical to the standalone JSON.

Both current files decode as strict UTF-8 without BOM or CR, have exactly one
terminal LF, and the JSON has no duplicate key, non-finite value, or negative
zero. All object keys are recursively UTF-8 lexical. All 55 amendment-bound
paths match: 3/3 runtime-bootstrap files, 24/24 TempoRAC source files, 19/19
tests, and 9/9 planning/review files. No input drift was observed.

I fully reread the project-local `research-refine`, `research-review`, and
`experiment-audit` skills and their reviewer-routing, assurance, independence,
trace, output, scope, integration, and experiment-integrity protocols. I also
reread the complete amendment pair, Round-3 review, canonical proposal, plan,
tracker, cited postfix pair, all 55 bound files, and relevant prior review
artifacts. Author summaries and intermediate candidate bytes were not used as
evidence.

## Blocking finding A005-R4-B1 - the native CPython initialization state is not closed

The launcher is a real pre-import native boundary and
`PySys_AddAuditHook` may validly be registered before interpreter
initialization. The problem is the state passed to that initialization.

JSON lines 853-903 call a 23-field JSON object the exact `PyConfig`. The public
CPython 3.12 `PyConfig` structure also contains authority-relevant inputs and
outputs that are absent here, including `use_frozen_modules`,
`filesystem_encoding`, `filesystem_errors`, `stdio_encoding`, `stdio_errors`,
`orig_argv`, `xoptions`, `warnoptions`, `int_max_str_digits`,
`pathconfig_warnings`, `platlibdir`, `pythonpath_env`, `stdlib_dir`, prefixes,
and the initialization-phase fields. The amendment also names neither
`PyConfig_InitIsolatedConfig` nor `PyConfig_InitPythonConfig`, and contains no
`PyPreConfig` at all. CPython preconfiguration separately controls allocator,
locale configuration/coercion, UTF-8 mode, environment use, and argv parsing.
Those choices occur before the hook-observed import sequence. The normative
API and field set are documented by the official
[CPython 3.12 initialization configuration](https://docs.python.org/3.12/c-api/init_config.html)
and the exact
[v3.12.13 `initconfig.h`](https://raw.githubusercontent.com/python/cpython/v3.12.13/Include/cpython/initconfig.h).

This is observable, not cosmetic. A supervisor can initialize the listed
fields from isolated defaults; another can initialize from Python defaults,
set the same listed fields, and choose a different unlisted
`use_frozen_modules` or preconfiguration. Both serialize exactly the mandated
23-key object. They can select frozen versus source origins and different
encoding/path state, changing the pre-import event sequence and potentially
the generated-source sequence. The exact supervisor binary merely freezes the
uncontracted choice made by that implementation; the verifier has no rule that
makes one choice conforming and the other nonconforming.

The same gap exists one level lower in JSON lines 815-840. Each inherited FD
row supplies only `fd`, `mode`, and a prose payload. Descriptor type, open
flags, current offset, seekability/sealing, exact byte length, and the
treatment of descriptors 1, 2, and all extras are not frozen. Those properties
are visible to CPython while it constructs standard streams because
`configure_c_stdio=1`. Two read-only descriptors containing the same bytes do
not constitute one exact launch state.

Minimal repair:

- freeze the exact preinitialization and initialization call sequence,
  including the specific `PyPreConfig_Init*` and `PyConfig_Init*` constructors;
- either close every platform-applicable CPython 3.12 input field and required
  output/readback field, or normatively say that every unlisted field remains
  the exact value produced by one named constructor and verify the complete
  resulting configuration before import;
- explicitly bind at least frozen-module selection, filesystem/stdio encoding,
  argv/xoptions/warnoptions, integer-string limit, path inputs/outputs, locale,
  UTF-8 mode, allocator, and main/importlib phase; and
- define one closed FD-row schema with descriptor class, flags, initial offset,
  length/sealing and EOF rules, exact handling of 1/2, and rejection/closure of
  every unlisted descriptor. State when the supervisor copies, resets, or
  closes each descriptor before Python initialization.

## Blocking finding A005-R4-B2 - generated-source discovery depends on its own final precommit

The desired authority order is sound as an abstract DAG: source, installed,
and CPython indexes precede generated-source rows; those rows precede the
environment; post-run traces come last. The specified execution used to obtain
the generated rows does not follow that order.

JSON line 942 says P06 runs the import-only bootstrap twice, captures the two
generated-source sequences, and precommits the resulting index. Yet:

- the final CPU environment contains
  `trusted_generated_source_index_sha256` and the final native launcher;
- every P06 owner is one of the exact 57 environment-bound supervisor
  entrypoints;
- JSON lines 848-851 require the controller to validate the generated index
  and environment before `exec`, and require the supervisor to consume all
  already trusted generated rows before `DISPATCH_BEGIN`;
- JSON lines 1135 and 1453 say every entrypoint reaches the one exact
  `protocol_entry.dispatch`; while line 942 says the discovery executions run
  without dispatch and without reading its input; and
- the fixed argv, envp, and FD table define no discovery-mode selector or
  pre-environment launch capsule.

Therefore a discovery process either starts under the final environment and
must already know the index it is supposed to discover, or starts under an
uncommitted environment/launcher state not covered by any closed schema. In
the former case the construction is circular. In the latter case equality of
two runs does not prove that the produced bytes are a function of the accepted
contract/runtime, especially given A005-R4-B1, and the P06 trace cannot
truthfully bind the final environment digest. Treating those runs as children
does not solve the problem: child argv/envp/FD roles are the same, and no exact
authenticated bit distinguishes bootstrap-only capture from normal dispatch.

Minimal repair: within the existing P06 owner and without adding a job, define
an explicit two-phase construction. The discovery phase needs a closed,
acyclic launch capsule that binds the effective contract, supervisor,
preconfiguration/configuration, source, wheel, installed, and CPython ledgers
but deliberately contains no generated-index, final-environment, or trace
digest; it also needs an exact authenticated bootstrap-only mode and event
schema. After two equal discoveries create the generated index and final
environment, a distinct final P06 verification invocation under those final
bytes must consume the entire index, run the ordinary dispatch/receipt path,
and bind the final trace. Alternatively, define a deterministic static
generated-source derivation that requires no execution. In either design, no
discovery result may authorize the process that generated it.

## Blocking finding A005-R4-B3 - member and trace associations lack unique typed preimages

Two new schemas contain the necessary logical information but do not specify
one byte representation or independently checkable association.

First, installed-distribution rows have both `normalized_member_path` and
`wheel_member_path_or_null` (JSON lines 713-766). The normalization rule says
the former is a wheel-member path, but never requires it to equal the
normalized form of a non-null latter value. For an installer-generated null
row there is, by definition, no wheel-member path, yet the former remains a
required ordering field with no derivation. A row can therefore order and
deduplicate under one path while claiming a different wheel member, and null
rows can choose arbitrary otherwise-valid ordering tokens. The stated
one-to-one wheel/member coverage is not mechanically decidable from this
schema.

Second, every executable-origin event has only five keys:
`event`, `origin`, `process_ordinal`, `sequence`, and `sha256` (JSON lines
1079-1085). The normalization clauses at lines 1121-1129 describe composites
such as path plus build ID, builtin/frozen token plus provider path, wheel
distribution plus installed path, and generated profile/ordinal/source bytes
plus generator digest/qualname/filename/mode. They do not state whether
`origin` is a string, object, or byte encoding, give delimiters/length prefixes,
or provide event-specific closed keys. In the generated case the five-key row
cannot independently demonstrate all values that line 1100 requires the
verifier to compare. Different encodings hash to different trace bytes, and a
digest-only row can hide a generator-origin or compile-mode substitution.

Minimal repair:

- for every non-null installed row require
  `normalized_member_path == normalize(wheel_member_path_or_null)` and bind the
  exact archive member; for a null row define one collision-free derived token
  from distribution, kind, and installed path (or make both member fields
  null and order on an explicitly defined generated-row key); and freeze the
  allowed installer-generated owner/byte derivation; and
- replace the polymorphic trace `origin` with exact event-specific closed
  objects, or define one tagged length-prefixed encoding for every event token.
  Persist all fields required for independent membership checking, including
  native build ID, provider path, distribution/member identity, and the full
  trusted-generator row identity. Require parse/re-encode byte equality before
  accepting a trace hash.

## R3-B2 and regression attack results

The Round-3 typed-DAG blocker is closed in this candidate.

- The inventory has exactly 54 unique owners with class counts
  `17+27+1+1+3+1+1+3=54`.
- The exact roster contains 106 upstream occurrences. Independent counting
  reproduced the declared decomposition: 58 generic stage, 30 run, 3 G1,
  2 K1, 3 X0I, 3 K3, 4 K4, and 3 NATP.
- Every roster dependency names an earlier unique owner; a topological walk
  has no forward, missing, duplicate-owner, or cyclic roster edge.
- NATP has exactly fourteen closed keys. Its ordered length-one
  `upstream_receipt_sha256[0]` is required to equal both
  `k6_receipt_sha256` and the unique `K6-RESAMPLER` receipt. The three NATP
  owners, their six 1,608-row seed indexes, 3,216 prediction edges per seed,
  two 4,824-row condition merges, pilot scope, and final 9,648-row product are
  root-bound.
- `k1.upstream` is a legal role. Ordinal 0 is exactly G0 and is checked through
  the exact G0 P402 payload index; ordinal 1 is exactly G1 and is checked
  through the selection/tune/score/checkpoint/three-run evidence. K1's own
  nine-key receipt remains unchanged.
- The exact 57 entrypoint descriptors begin with the same 54 owner tokens and
  end only with G5A-COMMIT, G5B-JOIN, and K7-FINAL.
- The G5a root index has exactly twelve ASCII-ordered classes, including the
  executable-origin, NATP completion, and pre-G5a inventory rows. Its root
  exclusions remain non-self-referential.

A recursive comparison against the archived Round-3 JSON found changes only
in round/version/lineage/bindings, the intended CPU-runtime/executable-closure
repair, the intended NATP/K1 inventory/DAG repair, test-contract text, and
disposition. The score, checkpoint, tune-input/output, natural inference,
K1/Cdev, target, prediction, capability/vault order, F4 Mapping composition,
science, job count, resource, data, threshold, and claim objects did not drift.

Consequently the prior status is:

| Item | Round-4 assessment |
|---|---|
| A005-R3-B1 | **REVISE** - the added ledgers and pre-import supervisor are substantial, but A005-R4-B1 through B3 prevent executable byte closure |
| A005-R3-B2 | **CLOSED** - NATP exact14, ordered K6 equality, two K1 roles, 54 owners, and 106 backward direct edges are coherent |
| A005-R2-B1 | **CLOSED** - exact normalized SmoothL1, one-run bounds, ten traversal ledgers, three-run order, and selection replay are unchanged |
| A005-R2-B2 | **SPECIFICATION CLOSED; END-TO-END BLOCKED** - FE/MXCSR/FTZ/DAZ/denormal/backend controls remain exact, but rely on the unresolved native launch |
| A005-R2-B3 | **OPEN via R4-B1/B2/B3** - the post-run direction has no hash feedback, but its producing runtime and origin encodings are not closed |
| A005-R2-B4/B5/B6/B7/B8 | **CLOSED** - inventory, direct run/G1/K1 lineage, NATP scope, K1 row preimage, and current bindings passed |
| F4 | **CLOSED** - snapshot-before-validation direct `MappingProxyType` composition is unchanged |

The intended 16-member checkpoint, 120-checkpoint/tune-output/evaluation grid,
shared 56-view input, binary64 reduction and deterministic argmin, complete
loser retention, one global K7 selection replay, 402-row immutable K1 outcome,
exact val/CERTIFIED Cdev projection with 108/8 floors, seven-member target,
ten-key target receipt, 9,648 manifest-derived component envelopes, and
preconsume -> atomic consume -> G5b -> K7 order all remain present. No free
score scalar, winner-only artifact path, caller component relabel, late Cdev
filter, preconsume vault access, or direct-Mapping TOCTOU path was reintroduced.

## Implementation, scope, and authority ceiling

The current source and tests are a non-authorizing snapshot and do not yet
implement the proposed supervisor, ledgers, schemas, or runtime controls. This
review did not run a gate-bearing test suite or any experiment. It accessed no
server, natural data, sealed/heldout material, evaluator vault, credential,
GPU, checkpoint, prediction, target, training entrypoint, or result. It changed
no amendment, proposal, plan, tracker, MANIFEST, source, test, paper, Git,
server, data, or experiment artifact.

The required repairs are provenance/serialization clarifications within the
existing P06 and runtime boundary. They must not add an experiment job, model,
algorithm, loss, parameter, seed, identity, dataset row, GPU hour, metric,
threshold, target/G5a receipt key, result, or claim.

This same-family review is provisional and non-authoritative. It authorizes
nothing: `P0=0`, `P1=0`, `P2=0`, `P2-METRIC=0`, `P3=0`, `S0=0`, `gate=0`,
`capability=0`, `evaluator=0`, `launch=0`, `server=0`, `data=0`, `GPU=0`,
`training=0`, `test=0`, `result=0`, `paper/claim=0`, and `Git=0`.

Final disposition: `REVISE`. The exact `_20260816_154812` pair with hashes
`0552aaca74b34929ead1124cce846d2a583931aa89ad29fcf0f97486b7be0198`
and `04d24ad19b19470bcd675974766a279dc7691d3fdaecf665e3fd24e4a4aaeb82`
cannot form an accepted effective contract. A newly versioned repair pair must
receive another normative review before any separately reviewed implementation
amendment; every execution and launch barrier remains zero.
