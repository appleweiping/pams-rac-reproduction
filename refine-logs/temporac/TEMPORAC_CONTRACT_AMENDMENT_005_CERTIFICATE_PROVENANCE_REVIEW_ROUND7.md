# TempoRAC Contract Amendment 005 Certificate Provenance - Round 7 Normative Review

Date: 2026-08-17  
Reviewer: `/root/temporac_certificate_amendment005_review`  
Route: Codex `gpt-5.6-sol`, OpenAI same-family, ultra review tier  
Assurance: provisional; not cross-family independent  
Verdict: **REVISE**  
Blocking groups: **6**  
Authority: **none**

## Outcome

The exact `_20260816_215501` Amendment 005 pair is not yet an executable,
substitution-closed provenance specification. It substantially closes the
native-build, launch, ledger and trace design attempted after Round 6, but six
blocking gaps remain:

1. FD6 `FIONREAD` values are required to stay equal to a zero-queued launch
   row after the child has already emitted frames, without an acknowledgement
   protocol; success evidence is therefore scheduler-dependent.
2. Durable publication permits content-addressed `EEXIST` but does not define
   the corresponding temp-file cleanup or a crash-recovery state machine.
3. all 57 dispatch descriptors name an undefined
   `temporac.stage-input-index.v4` schema.
4. trusted generated-source rows omit the exact executable code-object
   authority observed by the adjacent `exec` event.
5. before/after executable-map snapshots cannot observe an ELF or executable
   mapping loaded, executed and unloaded entirely inside a boundary.
6. the raw-audit catalog can name argument-projection schemas for ignored
   events, but the contract defines projections only for compile, exec, import
   and `ctypes.dlopen`.

These are normative implementability and authority-closure defects, not
requests to broaden the method. The verdict is `REVISE`. No Round-7 successor
effective-contract index may be constructed from this review, and the active
effective contract remains the accepted Round-6 index.

## Reviewed bytes and terminal integrity

The exact inputs matched at the opening snapshot and again after review
persistence:

| Input | Bytes | SHA-256 |
|---|---:|---|
| timestamped Round-7 Markdown `_20260816_215501.md` | 629,249 | `217ad51f4ea02f511777a63f0158f8f2864f4a786e201fa06cadbb5872ece751` |
| timestamped Round-7 JSON `_20260816_215501.json` | 582,615 | `31c6ad83f61214b1f49b83321116bbe53ec8eb2fb5c463afd552e0ace418de17` |
| fixed Markdown alias | 629,249 | `217ad51f4ea02f511777a63f0158f8f2864f4a786e201fa06cadbb5872ece751` |
| fixed JSON alias | 582,615 | `31c6ad83f61214b1f49b83321116bbe53ec8eb2fb5c463afd552e0ace418de17` |
| archived accepted Round-6 Markdown `_20260816_215500.md` | 312,728 | `2eaa3cfb746804fdc5890fec4c252c877a031a4abedef7b9ef6eeac0a4839395` |
| archived accepted Round-6 JSON `_20260816_215500.json` | 261,196 | `638efc8708f74c8514247f4f4c5e2311595e90edc6744c9b371886f0789af164` |
| Round-6 review Markdown | 15,666 | `4a3199c3ac18e3fd2a5bb7ecf6d66e10bc4a5cebb82363803c47867234b70e67` |
| Round-6 review JSON | 10,036 | `fcf8a9d4949f6b456f18bfb4591c512b6628c6ec7eb5a055873a995992f46ad6` |
| canonical proposal | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| experiment plan | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| experiment tracker | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |

The fixed aliases are byte-identical to the timestamped pair, and the
`_20260816_215500` archive retains the Round-6 bytes. Markdown Appendix A is
byte-identical to the standalone Round-7 JSON. Both current files are strict
UTF-8 without BOM or CR and have exactly one terminal LF. Duplicate-key
rejecting parsing succeeds; keys are recursively UTF-8 bytewise ordered and no
non-finite number or negative zero occurs.

All 70 amendment-declared current bindings matched at both snapshots: 4
runtime/bootstrap files, 24 source files, 19 tests and 23 planning/review
files. All 37 lineage path/hash records matched at the opening snapshot. At
terminal rehash, 36 immutable lineage records still matched; the sole expected
difference was the authorized same-run trace aggregator advancing from the
bound Round-6 hash
`7a860dc5a9be3878aab060e541430a5731237fc1b2e9b75d7798100a0a3a1da2`
to the turn-007 run-meta hash. No unexpected or concurrent bound-input drift
was observed.

I fully reread the project-local `research-refine`, `research-review` and
`experiment-audit` skills and every directly required routing, assurance,
independence, tracing, output, scope, integration and experiment-integrity
protocol. I then reread the complete current amendment MD/JSON, Round-6 review
pair and trace, proposal, plan, tracker, all current bindings, all cited
prerequisite reviews/amendments, the four runtime/bootstrap files, all 24
TempoRAC source files and all 19 TempoRAC tests. Author prechecks and summaries
were not accepted as evidence.

## A005-R7-B1 - blocking: FD6 queued-byte equality is nondeterministic

The preserved FD rule at JSON line 1312 fixes the pipe's `FIONREAD` queued
bytes to zero at entry. JSON line 7866 then requires a fresh `FIONREAD` value
at every operational stage and requires each present row to equal the launch
row. Yet the exact success order at lines 6705-6715 sends carrier, process,
pre-initial-image, mapping and FD evidence frames on FD6 before the later
operational readbacks; line 4908 confirms that these early rows are emitted
directly. The twin rule at line 6994 additionally requires the three stage
projections to be byte-identical after zeroing only the launch digest.

FD6 is a one-way pipe and the protocol defines no reverse acknowledgement or
drain barrier. After a write, `FIONREAD` reports a value determined by how far
the controller has drained the pipe. It may be zero or nonzero at each child
snapshot, and two fresh children need not observe the same value. Polling until
zero would add an unbounded synchronization operation that is not in the exact
state machine. Thus conforming implementations cannot deterministically
produce the required byte equality.

Minimal repair: choose and freeze one closed rule. Either require zero only at
physical entry and retain actual later nonnegative values while excluding
`queued_bytes` from launch/twin equality; buffer all early success frames until
after the last FD snapshot; or define an explicit controller acknowledgement
protocol, its FD/layout, framing, time/order bounds and failure paths. Add slow
reader, fast reader and twin-scheduling adversarial tests.

This keeps R7-B1 and R7-B2 open; it does not weaken the otherwise sound FD7
sealed carrier, FD8 ledger carrier or three-key failure framing.

## A005-R7-B2 - blocking: durable `EEXIST` and crash recovery are incomplete

JSON lines 6340-6341 allow a content-addressed immutable object, and in one
case a byte-identical prelaunch wrapper, already to exist. The only concrete
write algorithm at line 6360 always creates deterministic
`.tmp.<pid>.<starttime>` bytes and invokes `renameat2(RENAME_NOREPLACE)`. It
does not define a final-exists fast path or what happens to the completed temp
file when rename returns `EEXIST`: there is no winner validation branch,
`unlinkat(temp)`, directory re-fsync or proof that the temp is absent.

The same algorithm has no state lattice for crashes after request publication,
temp creation, data sync, rename or directory sync. Line 6359 rejects unrelated
or unresolvable residual entries on every later request, while line 6355 makes
the attempt nonretryable. A permitted `EEXIST` race or ordinary crash can
therefore leave a temp/request prefix that permanently poisons the durable
root, unless an implementation invents cleanup or global fail-stop semantics.
The object-store construction reuses this algorithm, so the ambiguity also
infects native-build evidence publication.

Minimal repair: freeze a single-writer/locking rule and an exact durable-state
lattice from request through attempt, prelaunch, joined objects and
attestation. Define a validated final-exists branch. On permitted rename
`EEXIST`, validate the winner, remove the exact temp, fsync the directory and
prove absence. Define strictly identified, idempotent stale-temp recovery for
every crash point, or explicitly make the whole deployment root irrecoverably
fail-stop and state the required redeployment procedure. Add crash-injection
tests at every open/write/file-sync/rename/directory-sync/reopen boundary.

## A005-R7-B3 - blocking: the 57-owner input-index ABI has no schema

The mechanical descriptor constructor at JSON line 6533 assigns every one of
the 57 owners `input_index_schema = temporac.stage-input-index.v4`; lines
6568-6571 require invocation bytes to parse as that exact schema. An exact
search across the bound proposal, amendment, source and tests finds no
definition outside this Amendment-005 token: no top-level or row closed keys,
canonical preimage, owner/handler identity, role universe, row ordering,
content bindings, multiplicities or per-owner override table exists.

Consequently `_dispatch_exact_owner` cannot be implemented from the contract,
and a validly encoded input index for one owner cannot be mechanically rejected
when substituted for another. This affects all 54 pre-G5a owners and the
G5A-COMMIT, G5B-JOIN and K7-FINAL suffix, including the internal K7
continuation.

Minimal repair: define one complete canonical
`temporac.stage-input-index.v4` with an exhaustive owner-specific mapping of
allowed roles, content-addressed inputs, order and multiplicity, and bind
owner plus handler token in the preimage. Alternatively, replace the common
token with exact already-defined owner-specific schemas through a closed
57-row override table. Add missing/extra/reordered, wrong-owner/handler,
cross-owner-valid-schema and G5A/G5B/K7 substitution tests.

## A005-R7-B4 - blocking: generated source does not bind generated code

The raw `exec` projection at JSON lines 4470-4481 correctly contains
`code_authority_sha256`, which hashes the exact code-object projection. The
generated discovery row at lines 7124-7151 has only eleven source/generator
fields; the trusted row at lines 7154-7170 adds only
`trusted_row_sha256`. The trusted-generated origin's exact key set and
one-to-one projection at lines 5260-5272 and 5295-5308 also omit code
authority.

Thus twin output equality, trusted-row hashing and full-run consumption bind
the source bytes and filename but not the code object that the adjacent
`exec` actually executes. The same source and filename can be compiled with a
different mode, flags, optimization or `dont_inherit` state, or an adjacent
unrelated/preexisting code object can be consumed. The raw trace records the
actual code digest but has no precommitted expected value against which to
compare it.

Minimal repair: add `exec_code_authority_sha256`, or the exact adjacent
`exec`-argument digest, to the discovery output, trusted-row preimage,
trusted-generated origin/projection, twin equality and full-run consumption.
Populate it only from the immediately adjacent exec whose source authority
equals the generated source hash. Update all exact key counts/preimages and
add mode, flags, optimization, inherited-future-flags and unrelated-code
substitution tests.

## A005-R7-B5 - blocking: boundary snapshots miss transient native execution

JSON line 4814 observes executable mappings only before and after
`Py_InitializeFromConfig` or one top-level `PyImport_ImportModule`; line 4901
replays those deltas and compares a terminal snapshot. This proves persistent
additions/removals at the chosen barriers, but it cannot observe a native
extension or dependency that calls `dlopen`/`dlmopen`, executes constructors
or code, and calls `dlclose` before the after snapshot. The same issue applies
to executable `mmap`/`mprotect` followed by `munmap` within the boundary.

The official CPython 3.12.13 audit table is complete for CPython/stdlib audit
calls and exposes Python-level `ctypes.dlopen`, not arbitrary native
`dlopen`/`dlmopen`/`dlclose` transitions. The Linux dynamic-loader interface
can recursively load dependencies, execute constructors before `dlopen`
returns, execute destructors before `dlclose` returns and unload mappings.
Therefore `before == after` can pass after unledgered executable code ran, in
conflict with the contract's claim of zero unaccounted executable mappings.
See the [CPython 3.12 audit-event table](https://docs.python.org/3.12/library/audit_events.html),
[glibc dynamic-linker hardening guidance](https://sourceware.org/glibc/manual/2.43/html_node/Dynamic-Linker-Hardening.html),
and the [Linux dlopen/dlmopen/dlclose manual](https://man7.org/linux/man-pages/man3/dlmopen.3.html).

Minimal repair: bind an immutable loader/executable-map transition mechanism
that observes native `dlopen`, `dlmopen`, `dlclose`, executable
`mmap`/`mprotect` and `munmap` before code can run, with an exact event-to-ELF
and provider bijection. If an interposition/audit/ptrace/seccomp design is
chosen, freeze its own executable origin, ordering, recursion and failure
semantics. A different acceptable route is a complete static/binary proof over
every reachable ELF that loader and executable-map APIs are absent/unreachable,
combined with binding-now/no-unload policy and negative tests. Boundary
snapshots alone are insufficient.

## A005-R7-B6 - blocking: raw audit argument projections are not closed

`audit_and_trace.argument_projections` at JSON lines 4451-4590 defines only
four schemas: compile, `ctypes.dlopen`, exec and import. The catalog rules at
lines 4656-4673 require every ignored-nonexecutive event to be explicitly
enumerated and selected by `arg_projection_schema`; the raw serialization rule
at lines 5313-5322 says `args` is the event-specific closed typed projection
named by that catalog row.

Initialization, import and finalization necessarily retain other raw audit
events such as `open`, `os.listdir` and `mmap.__new__`, but neither the
seven-key catalog row nor any BASE-bound registry defines those projection
objects' keys, Python type discriminants or canonical value rules. A catalog
can therefore name an undefined schema token. Independent producers and
verifiers cannot derive the same bytes or reject cross-schema/type
substitution.

Minimal repair: make a complete projection registry part of BASE authority,
or define each permitted projection literally. Freeze an exact
event-name/arity/stage/action-to-schema bijection, closed keys, exact Python
types and canonical values, including unsupported-type failure. Add tests for
missing/extra fields, arity/type/stage changes, cross-schema swaps and every
ignored event in initialization/import/finalization.

## Closure and regression attack

The six blockers above are the complete blocker set found in this review. The
remaining requested surfaces were mechanically coherent:

- FD7 retains writer-open populate/hash/seal, exact `/proc/self/fd` O_RDONLY
  reopen to a distinct OFD, writer close/`EBADF`, reader revalidation and exact
  child handoff. FD8 binds the correct ledger bytes. BASE has exactly 19 unique
  roles; FINAL has exactly 24 and begins with the byte-identical BASE19 prefix.
  Failure framing has exactly 45 unique stages (33 child and 12 controller)
  with coherent terminal/EOF/exit partitions.
- PyPreConfig still has exactly 10 fields and PyConfig exactly 64, with the
  prior constructor/setter/readback/error/clear closure. The 3.12.4 historical
  science environment and the stock 3.12.13 provenance verifier are separate
  typed version layers; the latter cannot relabel historical artifacts.
- The dispatch owner order is 57 unique tokens. Its first 54 are byte-equal to
  the pre-G5a roster order and its suffix is exactly G5A-COMMIT, G5B-JOIN,
  K7-FINAL. The current 54-node/106-edge DAG is correctly described only as the
  pre-G5a roster-direct projection, not as the future full runtime DAG. It is
  acyclic; every edge points backward, NATP is exact fourteen-key and points to
  K6, and K1 upstream ordinal 0/1 remains G0/G1.
- The normalized executable-event universe is exactly 16 unique tokens; the
  origin union is exactly 13 unique kinds and its event maps cover the 16
  tokens. Apart from B5/B6, origin preimages, typed installed/repository/native
  associations, raw/normalized row hashes and topological ledger ordering are
  noncircular.
- The CPython 3.12.13 source archive pin is 20,801,708 bytes with SHA-256
  `c08bc65a81971c1dd5783182826503369466c7e67374d1646519adf05207b684`.
  The Hatchling 1.27.0 wheel pin is
  `d3a2f3567c4f926ea39849cdf924c7e99e6686c9c8e288ae1037c8fa2a5d937b`.
  The `SOURCE_DATE_EPOCH=315532800` recipe, twin input/output/temp/stream
  equality, generated-exec evidence, depfile closure, wheel member-to-install
  association, native install, ELF/build-id/DT_NEEDED/RPATH/provider rows and
  no-wheel native members have no additional static blocker. These future
  builds were not executed.
- The original checkpoint/tune closure is unchanged: 16 members per
  checkpoint, exact 56-view tune input, all 120 checkpoints/raw outputs/eval
  receipts/score rows, normalized SmoothL1 formula, one-run bounds, three-run
  order, binary64/Neumaier reduction, strict CPU load/forward and deterministic
  selection remain byte-bound with no free scalar.
- K1 still commits the immutable 402-row outcome/index before G5a. Cdev remains
  exactly the 134-row val and CERTIFIED projection with floors 108/8, not all
  402. All 9,648 envelopes bind the canonical population component token. The
  seven-member target, unchanged ten-key target receipt, G5a root, G5b join and
  K7 bytewise rederivation retain their schemas and order.
- Capability order remains preconsume non-vault checks, atomic consume, G5b
  vault validation/join, then K7. F4 remains snapshot-before-validation and
  direct immutable `MappingProxyType` composition. Data firewall, UTF-8,
  duplicate-key rejection and failure-closed byte comparison remain intact.
- No algorithm, loss, model, trainable component, job, seed, arm, condition,
  dataset identity, compute hour, threshold, metric, result, claim or
  paper-visible block changed. The plan remains 27 training jobs, 93 planned
  A6000 hours under the 100-hour ceiling.

The intended-closure matrix is:

| Intended item | Round-7 assessment |
|---|---|
| R7-B1 | **OPEN** - FD6 post-write `FIONREAD` equality is scheduler-dependent (A005-R7-B1) |
| R7-B2 | **OPEN** - the same FD6 contradiction prevents deterministic universal success transport (A005-R7-B1) |
| R7-B3 | **OPEN** - `EEXIST`/temp cleanup and crash recovery are not an executable durable state machine (A005-R7-B2) |
| R7-B4 | **CLOSED** - BASE19 to FINAL24 byte carriers, role order and topological exclusions are coherent |
| R7-B5 | **OPEN** - all 57 descriptors refer to an undefined input-index schema (A005-R7-B3) |
| R7-B6 | **CLOSED** - 3.12.4 science and 3.12.13 provenance remain distinctly typed |
| R7-B7 | **OPEN** - trusted generated source omits adjacent executable code authority (A005-R7-B4) |
| R7-B8 | **OPEN** - native build structure is coherent, but transient native execution escapes its mapping evidence (A005-R7-B5) |
| R7-B9 | **OPEN** - transient loader activity and undefined raw argument projections defeat complete trace closure (A005-R7-B5/B6) |
| Round-6 scientific/receipt closures | **NO REGRESSION** |
| Science/jobs/data/claims | **UNCHANGED** |

## Effective-contract lineage, implementation and authority ceiling

The active effective-contract index reconstructs independently to exactly 665
bytes and SHA-256
`c203f4aab00879acf041b90829f0124d0b1de231294f191b03f3166f728e3d5c`.
Its five rows are the canonical proposal, archived/current accepted Round-6
amendment MD/JSON and Round-6 ACCEPT review MD/JSON. It is acyclic and
non-self-hashing.

The prospective Round-7 five-row recipe is also structurally acyclic, but its
last two roles expressly require an accepted review. This review is `REVISE`,
so no successor bytes or digest exist. The fixed aliases, rejected review,
archives and traces are not contract constituents, and no new runtime artifact
may bind a guessed Round-7 digest. The active Round-6 digest remains the only
effective contract digest.

The current source and tests do not implement this prospective native
supervisor/controller/build/trace ABI. This review ran only read-only byte,
schema, source, test and mechanical checks; it did not run a gate-bearing test
suite, native build, controller, discovery, replay, evaluator, training job or
experiment.

No amendment, proposal, plan, tracker, MANIFEST, source, test, paper, Git,
server, data, training or experiment artifact was modified. The only review
writes are the Round-7 review pair and its authorized continuation trace/event.

This same-family review is provisional and non-authoritative. `REVISE`
authorizes no implementation, successor contract, build, test or execution:
`P0=0`, `P1=0`, `P2=0`, `P2-METRIC=0`, `P3=0`, `S0=0`, `gate=0`,
`capability=0`, `evaluator=0`, `launch=0`, `server=0`, `data=0`, `sealed=0`,
`heldout=0`, `GPU=0`, `training=0`, `test=0`, `result=0`, `paper/claim=0`,
and `Git=0`.

Final disposition: `REVISE` for the exact 629,249-byte Markdown
`217ad51f4ea02f511777a63f0158f8f2864f4a786e201fa06cadbb5872ece751`
and 582,615-byte JSON
`31c6ad83f61214b1f49b83321116bbe53ec8eb2fb5c463afd552e0ace418de17`,
subject to same-family provisional assurance and the zero-authority ceiling
above.
